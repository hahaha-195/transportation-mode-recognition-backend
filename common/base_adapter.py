# common/base_adapter.py

import numpy as np
import pandas as pd
import pickle
import os
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
from abc import ABC, abstractmethod

from common.trajectory_cleaner import TrajectoryCleaner


class BaseDataAdapter(ABC):
    """
    数据适配器基类
    - 清洗结果全局共享，只按 cleaning_mode 区分
    - 各实验只做格式转换
    """

    # 所有实验共用的7类标签
    VALID_LABELS = {
        'Walk', 'Bike', 'Bus', 'Car & taxi',
        'Train', 'Subway', 'Airplane'
    }

    # 统一目标长度
    TARGET_LENGTH = 50

    # 清洗模式预设参数
    CLEANING_PRESETS = {
        'strict': {
            'max_time_gap': 180.0,
            'max_bearing_change': 120.0,
            'min_segment_length': 15,
            'max_outlier_ratio': 0.15,
            'enable_smoothing': True,
            'smoothing_window': 7
        },
        'balanced': {
            'max_time_gap': 300.0,
            'max_bearing_change': 150.0,
            'min_segment_length': 10,
            'max_outlier_ratio': 0.25,
            'enable_smoothing': True,
            'smoothing_window': 5
        },
        'gentle': {
            'max_time_gap': 600.0,
            'max_bearing_change': 180.0,
            'min_segment_length': 8,
            'max_outlier_ratio': 0.35,
            'enable_smoothing': False,
            'smoothing_window': 3
        }
    }

    FEATURE_COLUMNS = [
        'latitude', 'longitude', 'speed', 'acceleration',
        'bearing_change', 'distance', 'time_diff',
        'total_distance', 'total_time'
    ]

    def __init__(self,
                 enable_cleaning: bool = True,
                 cleaning_mode: str = 'balanced',
                 cache_dir: str = '../data/processed'):
        """
        初始化基础适配器

        Args:
            enable_cleaning: 是否启用清洗
            cleaning_mode: 清洗模式 ('strict', 'balanced', 'gentle')
            cache_dir: 清洗缓存目录
        """
        self.target_length = self.TARGET_LENGTH
        self.enable_cleaning = enable_cleaning
        self.cleaning_mode = cleaning_mode
        self.cache_dir = cache_dir

        # 加载清洗参数
        self.cleaning_params = self.CLEANING_PRESETS.get(
            cleaning_mode, self.CLEANING_PRESETS['balanced']
        )

        # 初始化清洗器
        self.cleaner = TrajectoryCleaner(**self.cleaning_params)

        # 统计信息
        self.cleaning_stats = {'before': {}, 'after': {}, 'cleaner': {}}

    @property
    @abstractmethod
    def experiment_name(self) -> str:
        """返回实验名称（仅用于日志）"""
        pass

    # ============ 缓存机制（全局共享）============

    def _get_cache_path(self) -> str:
        """
        缓存路径：只按 cleaning_mode 区分
        例如: cleaned_balanced.pkl
        """
        os.makedirs(self.cache_dir, exist_ok=True)
        return os.path.join(self.cache_dir, f"cleaned_{self.cleaning_mode}.pkl")

    def _load_from_cache(self, cache_path: str) -> Optional[List]:
        """从缓存加载清洗后的数据"""
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached = pickle.load(f)
                print(f"✅ 从缓存加载: {cache_path}")
                print(f"   数据量: {len(cached)} 条")
                return cached
            except Exception as e:
                print(f"⚠️ 缓存加载失败: {e}")
        return None

    def _save_to_cache(self, data: List, cache_path: str):
        """保存清洗后的数据到缓存"""
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            size_mb = os.path.getsize(cache_path) / 1024 / 1024
            print(f"✅ 已保存缓存: {cache_path} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"⚠️ 缓存保存失败: {e}")

    # ============ 清洗流程 ============

    def _stage1_basic_filter(self, base_segments: List[dict]) -> Tuple[List[Tuple], int]:
        """
        第一阶段：基础过滤（标签、长度、特征提取）

        Returns:
            (valid_segments, discarded_count)
            valid_segments: List of (trajectory, datetime_series, label)
        """
        valid_segments = []
        discarded = 0
        min_len = self.cleaning_params['min_segment_length']

        for seg in tqdm(base_segments, desc="[阶段1: 基础过滤]"):
            # 标签过滤
            if seg['label'] not in self.VALID_LABELS:
                discarded += 1
                continue

            # 长度过滤
            if len(seg.get('raw_points', [])) < min_len:
                discarded += 1
                continue

            # 特征提取
            points = seg['raw_points']
            for col in self.FEATURE_COLUMNS:
                if col not in points.columns:
                    points[col] = 0.0

            trajectory = points[self.FEATURE_COLUMNS].values.astype(np.float32)
            datetime_series = seg.get('datetime_series')
            label = seg['label']

            valid_segments.append((trajectory, datetime_series, label))

        return valid_segments, discarded

    def _stage2_deep_cleaning(self, valid_segments: List[Tuple]) -> Tuple[List[Tuple], int]:
        """
        第二阶段：深度清洗

        Returns:
            (cleaned_segments, discarded_count)
            cleaned_segments: List of (trajectory, datetime_series, label)
        """
        cleaned_segments = []
        discarded = 0

        for trajectory, datetime_series, label in tqdm(valid_segments, desc="[阶段2: 深度清洗]"):
            # 执行清洗
            cleaned_traj, is_valid = self.cleaner.clean_segment(trajectory, label)

            if not is_valid:
                discarded += 1
                continue

            # 长度规范化
            cleaned_traj = self.cleaner.normalize_sequence_length(
                cleaned_traj, self.target_length
            )

            # 同步时间序列
            if datetime_series is not None:
                datetime_series = self._normalize_time_series(datetime_series)

            cleaned_segments.append((cleaned_traj, datetime_series, label))

        return cleaned_segments, discarded

    def _normalize_time_series(self, datetime_series: pd.Series) -> pd.Series:
        """规范化时间序列长度到 target_length"""
        current_length = len(datetime_series)

        if current_length == self.target_length:
            return datetime_series.reset_index(drop=True)
        elif current_length > self.target_length:
            indices = np.linspace(0, current_length - 1, self.target_length, dtype=int)
            return datetime_series.iloc[indices].reset_index(drop=True)
        else:
            last_time = datetime_series.iloc[-1]
            padding = pd.Series([last_time] * (self.target_length - current_length))
            return pd.concat([datetime_series.reset_index(drop=True), padding],
                             ignore_index=True)

    def _finalize_without_cleaning(self, segments: List[Tuple]) -> List[Tuple]:
        """跳过清洗时的长度规范化"""
        finalized = []
        for trajectory, datetime_series, label in segments:
            # 轨迹长度规范化
            L = len(trajectory)
            if L != self.target_length:
                if L > self.target_length:
                    indices = np.linspace(0, L - 1, self.target_length, dtype=int)
                    trajectory = trajectory[indices]
                else:
                    padding = np.zeros((self.target_length - L, 9), dtype=np.float32)
                    trajectory = np.vstack([trajectory, padding])

            # 时间序列规范化
            if datetime_series is not None:
                datetime_series = self._normalize_time_series(datetime_series)

            finalized.append((trajectory, datetime_series, label))
        return finalized

    # ============ 主入口 ============

    def process_segments(self, base_segments: List[dict],
                         use_cache: bool = True) -> List:
        """
        处理轨迹段（带全局缓存）

        Args:
            base_segments: 基础数据段列表
            use_cache: 是否使用缓存

        Returns:
            格式化后的数据列表（由子类 _format_output 决定）
        """
        print(f"\n{'=' * 60}")
        print(f"{self.experiment_name} 数据适配")
        print(f"清洗模式: {self.cleaning_mode}, 目标长度: {self.target_length}")
        print(f"{'=' * 60}")

        cache_path = self._get_cache_path()

        # 尝试从缓存加载
        if use_cache:
            cached_data = self._load_from_cache(cache_path)
            if cached_data is not None:
                self._print_label_distribution(cached_data)
                return self._format_output(cached_data)

        # 记录统计
        total_segments = len(base_segments)
        self.cleaning_stats['before'] = {
            'total_segments': total_segments,
            'total_points': sum(len(seg.get('raw_points', [])) for seg in base_segments)
        }

        # 第一阶段：基础过滤
        print("\n第一阶段: 基础预处理...")
        valid_segments, stage1_discarded = self._stage1_basic_filter(base_segments)
        print(f"  结果: {total_segments} → {len(valid_segments)} (丢弃 {stage1_discarded})")

        # 第二阶段：深度清洗
        if self.enable_cleaning:
            print(f"\n第二阶段: 深度清洗 (模式: {self.cleaning_mode})...")
            cleaned_segments, stage2_discarded = self._stage2_deep_cleaning(valid_segments)
            print(f"  结果: {len(valid_segments)} → {len(cleaned_segments)} (丢弃 {stage2_discarded})")
        else:
            print("\n⚠️ 跳过第二阶段清洗")
            cleaned_segments = self._finalize_without_cleaning(valid_segments)
            stage2_discarded = 0

        # 更新统计
        self.cleaning_stats['after'] = {
            'valid_segments': len(cleaned_segments),
            'stage1_discarded': stage1_discarded,
            'stage2_discarded': stage2_discarded,
            'retention_rate': len(cleaned_segments) / max(total_segments, 1),
        }
        self.cleaning_stats['cleaner'] = self.cleaner.get_cleaning_stats()

        # 保存到缓存
        if use_cache and cleaned_segments:
            self._save_to_cache(cleaned_segments, cache_path)

        # 打印报告
        self.print_cleaning_summary()
        self._print_label_distribution(cleaned_segments)

        return self._format_output(cleaned_segments)

    @abstractmethod
    def _format_output(self, cleaned_segments: List[Tuple]) -> List:
        """格式化输出（子类实现）"""
        pass

    def _print_label_distribution(self, processed: List[Tuple]):
        """打印标签分布"""
        from collections import Counter
        labels = [item[2] for item in processed]
        counts = Counter(labels)
        print(f"\n📊 标签分布 (共 {len(processed)} 条):")
        for label in sorted(counts.keys()):
            pct = counts[label] / len(processed) * 100
            print(f"   {label:15s}: {counts[label]:5d} ({pct:5.1f}%)")

    def print_cleaning_summary(self):
        """打印清洗摘要"""
        print("\n" + "=" * 60)
        print(f"数据清洗摘要")
        print("=" * 60)

        before = self.cleaning_stats.get('before', {})
        after = self.cleaning_stats.get('after', {})
        cleaner = self.cleaning_stats.get('cleaner', {})

        print(f"\n📌 第一阶段 (基础预处理):")
        print(f"   输入轨迹段数: {before.get('total_segments', 0):,}")
        print(f"   基础过滤丢弃: {after.get('stage1_discarded', 0):,}")

        print(f"\n🛠️ 第二阶段 (深度清洗):")
        print(f"   深度清洗丢弃: {after.get('stage2_discarded', 0):,}")
        print(f"   最终保留段数: {after.get('valid_segments', 0):,}")
        print(f"   总体保留率: {after.get('retention_rate', 0):.2%}")

        print(f"\n🔧 清洗操作详情:")
        print(f"   物理异常修复: {cleaner.get('outliers_removed', 0):,} 个点")
        print(f"   时间间隔插值: {cleaner.get('points_interpolated', 0):,} 个点")
        print(f"   轨迹平滑优化: {cleaner.get('points_smoothed', 0):,} 个点")

        print("=" * 60 + "\n")

    def get_cleaning_stats(self) -> Dict:
        return self.cleaning_stats.copy()