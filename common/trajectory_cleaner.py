import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d


class TrajectoryCleaner:
    """
    改进的轨迹数据清洗器 - 针对GeoLife交通方式识别优化

    核心改进:
    1. 分层清洗策略：物理异常检测 → 统计异常检测 → 平滑优化
    2. 交通方式自适应阈值：不同交通工具使用不同的物理约束
    3. 保留率控制：确保数据质量的同时最大化样本保留
    4. 智能插值：基于运动模式的缺失值填充
    """

    def __init__(self,
                 # 交通方式特定的速度上限 (m/s)
                 speed_limits: Dict[str, float] = None,
                 # 交通方式特定的加速度上限 (m/s²)
                 accel_limits: Dict[str, float] = None,
                 # 其他通用参数
                 max_time_gap: float = 300.0,
                 max_bearing_change: float = 150.0,
                 min_segment_length: int = 10,
                 # 清洗强度控制
                 max_outlier_ratio: float = 0.25,  # 降低到25%
                 enable_smoothing: bool = True,
                 smoothing_window: int = 5):
        """
        初始化轨迹清洗器

        Args:
            speed_limits: 各交通方式的速度上限字典
            accel_limits: 各交通方式的加速度上限字典
            max_time_gap: 最大允许时间间隔 (秒)
            max_bearing_change: 最大允许方向变化 (度)
            min_segment_length: 最小轨迹段长度
            max_outlier_ratio: 最大异常点比例 (超过则丢弃整段)
            enable_smoothing: 是否启用平滑
            smoothing_window: 平滑窗口大小
        """
        # 默认速度上限 (基于真实交通工具物理特性)
        self.speed_limits = speed_limits or {
            'Walk': 2.5,  # 步行: 2.5 m/s (~9 km/h)
            'Bike': 8.0,  # 自行车: 8 m/s (~28.8 km/h)
            'Bus': 22.0,  # 公交: 22 m/s (~79.2 km/h)
            'Car & taxi': 35.0,  # 汽车: 35 m/s (~126 km/h)
            'Train': 45.0,  # 火车: 45 m/s (~162 km/h)
            'Subway': 30.0,  # 地铁: 30 m/s (~108 km/h)
            'Airplane': 150.0,  # 飞机: 150 m/s (~540 km/h)
        }

        # 默认加速度上限 (基于真实交通工具物理特性)
        self.accel_limits = accel_limits or {
            'Walk': 2.0,  # 步行: 2 m/s²
            'Bike': 3.0,  # 自行车: 3 m/s²
            'Bus': 4.0,  # 公交: 4 m/s²
            'Car & taxi': 5.0,  # 汽车: 5 m/s²
            'Train': 2.0,  # 火车: 2 m/s² (加速慢)
            'Subway': 3.0,  # 地铁: 3 m/s²
            'Airplane': 8.0,  # 飞机: 8 m/s²
        }

        self.max_time_gap = max_time_gap
        self.max_bearing_change = max_bearing_change
        self.min_segment_length = min_segment_length
        self.max_outlier_ratio = max_outlier_ratio
        self.enable_smoothing = enable_smoothing
        self.smoothing_window = smoothing_window

        # 统计信息
        self.cleaning_stats = {
            'total_segments': 0,
            'segments_kept': 0,
            'segments_discarded': 0,
            'total_points': 0,
            'outliers_removed': 0,
            'points_interpolated': 0,
            'points_smoothed': 0,
            'discard_reasons': {
                'too_short': 0,
                'too_many_outliers': 0,
                'invalid_after_cleaning': 0,
            }
        }

    def clean_segment(self, trajectory: np.ndarray, label: str) -> Tuple[np.ndarray, bool]:
        """
        清洗单个轨迹段 - 三阶段清洗策略

        阶段1: 物理异常检测与修复 (基于交通工具物理特性)
        阶段2: 统计异常检测与平滑 (基于数据分布)
        阶段3: 轨迹完整性检查与优化

        Args:
            trajectory: 轨迹特征矩阵 (n_points, 9)
                列索引: [lat, lon, speed, accel, bearing_change,
                         distance, time_diff, total_distance, total_time]
            label: 交通方式标签

        Returns:
            (cleaned_trajectory, is_valid): 清洗后的轨迹和有效性标志
        """
        self.cleaning_stats['total_segments'] += 1
        self.cleaning_stats['total_points'] += len(trajectory)

        # 预检查: 长度过短
        if len(trajectory) < self.min_segment_length:
            self.cleaning_stats['segments_discarded'] += 1
            self.cleaning_stats['discard_reasons']['too_short'] += 1
            return np.array([]), False

        cleaned = trajectory.copy()

        # ============ 阶段1: 物理异常检测与修复 ============
        cleaned, outlier_count = self._detect_and_fix_physical_outliers(cleaned, label)

        # 检查异常点比例
        outlier_ratio = outlier_count / len(trajectory) if len(trajectory) > 0 else 0
        if outlier_ratio > self.max_outlier_ratio:
            self.cleaning_stats['segments_discarded'] += 1
            self.cleaning_stats['discard_reasons']['too_many_outliers'] += 1
            return np.array([]), False

        # ============ 阶段2: 时间连续性处理 ============
        cleaned, interpolated = self._handle_time_gaps(cleaned)
        self.cleaning_stats['points_interpolated'] += interpolated

        # ============ 阶段3: 方向平滑与优化 ============
        if self.enable_smoothing:
            cleaned, smoothed = self._smooth_trajectory(cleaned, label)
            self.cleaning_stats['points_smoothed'] += smoothed

        # 最终检查
        if len(cleaned) < self.min_segment_length:
            self.cleaning_stats['segments_discarded'] += 1
            self.cleaning_stats['discard_reasons']['invalid_after_cleaning'] += 1
            return np.array([]), False

        # 成功清洗
        self.cleaning_stats['segments_kept'] += 1
        return cleaned, True

    def _detect_and_fix_physical_outliers(self, trajectory: np.ndarray,
                                          label: str) -> Tuple[np.ndarray, int]:
        """
        阶段1: 物理异常检测与修复

        策略:
        1. 识别明显违反物理规律的点 (速度/加速度超限)
        2. 识别NaN/Inf值
        3. 使用局部中值滤波修复 (保留轨迹形状)

        Returns:
            (cleaned_trajectory, outlier_count)
        """
        if len(trajectory) == 0:
            return trajectory, 0

        n_points = len(trajectory)
        cleaned = trajectory.copy()

        # 提取特征
        speed = cleaned[:, 2]
        accel = cleaned[:, 3]

        # 获取该交通方式的阈值
        max_speed = self.speed_limits.get(label, 50.0)
        max_accel = self.accel_limits.get(label, 10.0)

        # 识别异常点
        outlier_mask = np.zeros(n_points, dtype=bool)

        # 1. 速度异常
        speed_outliers = (speed < 0) | (speed > max_speed * 1.2)  # 允许20%超限
        outlier_mask |= speed_outliers

        # 2. 加速度异常
        accel_outliers = np.abs(accel) > max_accel * 1.5  # 允许50%超限
        outlier_mask |= accel_outliers

        # 3. NaN/Inf异常
        nan_inf_mask = (np.isnan(speed) | np.isinf(speed) |
                        np.isnan(accel) | np.isinf(accel))
        outlier_mask |= nan_inf_mask

        outlier_count = np.sum(outlier_mask)

        # 修复策略: 局部中值滤波
        if outlier_count > 0:
            for idx in np.where(outlier_mask)[0]:
                # 使用5点窗口计算中值
                window_start = max(0, idx - 2)
                window_end = min(n_points, idx + 3)

                # 只对速度、加速度、方向变化进行修复
                for col in [2, 3, 4]:
                    window_values = cleaned[window_start:window_end, col]
                    # 排除当前异常点
                    valid_values = np.delete(window_values,
                                             idx - window_start if idx - window_start < len(window_values) else -1)

                    if len(valid_values) > 0:
                        cleaned[idx, col] = np.median(valid_values)
                    else:
                        # 如果窗口内没有有效值，使用全局中值
                        global_values = cleaned[:, col]
                        valid_global = global_values[~np.isnan(global_values) & ~np.isinf(global_values)]
                        if len(valid_global) > 0:
                            cleaned[idx, col] = np.median(valid_global)
                        else:
                            cleaned[idx, col] = 0.0

        self.cleaning_stats['outliers_removed'] += outlier_count
        return cleaned, outlier_count

    def _handle_time_gaps(self, trajectory: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        阶段2: 时间连续性处理

        策略:
        1. 检测大时间间隔 (> max_time_gap)
        2. 小间隔 (<60s): 线性插值
        3. 大间隔 (>60s): 标记但不插值 (可能是真实停留)

        Returns:
            (cleaned_trajectory, points_interpolated)
        """
        if len(trajectory) < 2:
            return trajectory, 0

        time_diffs = trajectory[1:, 6]  # time_diff列
        large_gaps = np.where(time_diffs > self.max_time_gap)[0]

        if len(large_gaps) == 0:
            return trajectory, 0

        cleaned = trajectory.copy()
        points_interpolated = 0

        for gap_idx in large_gaps:
            gap_size = time_diffs[gap_idx]

            # 只对中等时间间隔插值 (10s - 60s)
            if 10 < gap_size <= 60:
                # 计算需要插值的点数 (每10秒1个点)
                num_points = min(int(gap_size / 10), 5)

                if num_points >= 2:
                    start_point = cleaned[gap_idx]
                    end_point = cleaned[gap_idx + 1]

                    # 线性插值所有特征
                    for i in range(1, num_points):
                        ratio = i / num_points
                        new_point = start_point + ratio * (end_point - start_point)
                        cleaned = np.insert(cleaned, gap_idx + i, new_point, axis=0)
                        points_interpolated += 1

            # 大间隔 (>60s): 不插值，保持原样

        return cleaned, points_interpolated

    def _smooth_trajectory(self, trajectory: np.ndarray,
                           label: str) -> Tuple[np.ndarray, int]:
        """
        阶段3: 轨迹平滑与优化

        策略:
        1. 对速度、加速度使用Savitzky-Golay滤波器平滑
        2. 对方向变化使用移动平均平滑
        3. 保留轨迹的整体趋势

        Returns:
            (smoothed_trajectory, points_smoothed)
        """
        if len(trajectory) < self.smoothing_window:
            return trajectory, 0

        cleaned = trajectory.copy()
        points_smoothed = 0

        try:
            # 1. 速度平滑 (Savitzky-Golay滤波)
            if len(cleaned) >= self.smoothing_window:
                speed_smoothed = savgol_filter(
                    cleaned[:, 2],
                    window_length=self.smoothing_window,
                    polyorder=2,
                    mode='nearest'
                )
                cleaned[:, 2] = speed_smoothed
                points_smoothed += len(cleaned)

            # 2. 加速度平滑
            if len(cleaned) >= self.smoothing_window:
                accel_smoothed = savgol_filter(
                    cleaned[:, 3],
                    window_length=self.smoothing_window,
                    polyorder=2,
                    mode='nearest'
                )
                cleaned[:, 3] = accel_smoothed

            # 3. 方向变化平滑 (移动平均)
            bearing_changes = cleaned[:, 4]
            max_change = self.max_bearing_change

            # 根据交通方式调整阈值
            if label in ['Walk', 'Bike']:
                max_change *= 0.8  # 步行/骑行允许更大的方向变化

            # 识别异常方向变化
            abnormal_mask = bearing_changes > max_change
            abnormal_indices = np.where(abnormal_mask)[0]

            # 使用移动平均平滑
            for idx in abnormal_indices:
                if idx > 0 and idx < len(cleaned) - 1:
                    window_start = max(0, idx - 1)
                    window_end = min(len(cleaned), idx + 2)
                    cleaned[idx, 4] = np.mean(cleaned[window_start:window_end, 4])

        except Exception as e:
            # 平滑失败，返回原始轨迹
            return trajectory, 0

        return cleaned, points_smoothed

    def normalize_sequence_length(self, trajectory: np.ndarray,
                                  target_length: int = 50) -> np.ndarray:
        """
        统一序列长度 - 使用智能重采样

        策略:
        1. 下采样: 使用等间隔采样保留关键点
        2. 上采样: 使用样条插值保持轨迹平滑

        Args:
            trajectory: 轨迹特征矩阵
            target_length: 目标长度

        Returns:
            统一长度后的轨迹
        """
        if len(trajectory) == 0:
            return np.zeros((target_length, 9), dtype=np.float32)

        if len(trajectory) == target_length:
            return trajectory

        current_length = len(trajectory)

        if current_length > target_length:
            # 下采样: 等间隔采样
            indices = np.linspace(0, current_length - 1, target_length, dtype=int)
            return trajectory[indices]

        else:
            # 上采样: 线性插值
            try:
                x_old = np.arange(current_length)
                x_new = np.linspace(0, current_length - 1, target_length)

                trajectory_resampled = np.zeros((target_length, 9), dtype=np.float32)

                for col in range(9):
                    f = interp1d(x_old, trajectory[:, col], kind='linear',
                                 fill_value='extrapolate')
                    trajectory_resampled[:, col] = f(x_new)

                return trajectory_resampled

            except Exception:
                # 插值失败，使用零填充
                padding = np.zeros((target_length - current_length, 9), dtype=np.float32)
                return np.vstack([trajectory, padding])

    def get_cleaning_stats(self) -> Dict:
        """获取清洗统计信息"""
        stats = self.cleaning_stats.copy()

        # 计算额外指标
        if stats['total_segments'] > 0:
            stats['retention_rate'] = stats['segments_kept'] / stats['total_segments']
        else:
            stats['retention_rate'] = 0.0

        if stats['total_points'] > 0:
            stats['outlier_rate'] = stats['outliers_removed'] / stats['total_points']
        else:
            stats['outlier_rate'] = 0.0

        return stats

    def reset_stats(self):
        """重置统计信息"""
        self.cleaning_stats = {
            'total_segments': 0,
            'segments_kept': 0,
            'segments_discarded': 0,
            'total_points': 0,
            'outliers_removed': 0,
            'points_interpolated': 0,
            'points_smoothed': 0,
            'discard_reasons': {
                'too_short': 0,
                'too_many_outliers': 0,
                'invalid_after_cleaning': 0,
            }
        }

    def print_cleaning_summary(self):
        """打印清洗摘要"""
        stats = self.get_cleaning_stats()

        print("\n" + "=" * 70)
        print("轨迹数据清洗摘要报告")
        print("=" * 70)

        print(f"\n📊 总体统计:")
        print(f"  处理轨迹段总数: {stats['total_segments']:,}")
        print(f"  保留轨迹段数量: {stats['segments_kept']:,}")
        print(f"  丢弃轨迹段数量: {stats['segments_discarded']:,}")
        print(f"  保留率: {stats['retention_rate']:.2%}")

        print(f"\n🔍 数据质量:")
        print(f"  总轨迹点数: {stats['total_points']:,}")
        print(f"  检测异常点数: {stats['outliers_removed']:,}")
        print(f"  异常点比例: {stats['outlier_rate']:.2%}")

        print(f"\n🛠️ 清洗操作:")
        print(f"  物理异常修复: {stats['outliers_removed']:,} 个点")
        print(f"  时间间隔插值: {stats['points_interpolated']:,} 个点")
        print(f"  轨迹平滑优化: {stats['points_smoothed']:,} 个点")

        print(f"\n❌ 丢弃原因:")
        for reason, count in stats['discard_reasons'].items():
            print(f"  {reason}: {count:,}")

        print("=" * 70 + "\n")