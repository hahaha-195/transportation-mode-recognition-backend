"""
通用GeoLife数据预处理器
一次性提取所有实验共用的基础数据，避免重复处理

输出数据结构:
{
    'user_id': str,
    'trajectory_id': str,
    'raw_points': DataFrame,  # 原始GPS点 + 9维特征
    'label': str,             # 归一化后的标签
    'start_time': datetime,
    'end_time': datetime,
    'datetime_series': Series # 时间序列（用于天气匹配）
}
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple
from tqdm import tqdm
import pickle
import warnings

warnings.filterwarnings('ignore')


class BaseGeoLifePreprocessor:
    """GeoLife基础数据预处理器（所有实验共用）"""

    def __init__(self, geolife_root: str):
        self.geolife_root = geolife_root

        # 标签映射规则（统一所有实验）
        self.label_mapping = {
            'taxi': 'Car & taxi',
            'car': 'Car & taxi',
            'drive': 'Car & taxi',
            'bus': 'Bus',
            'walk': 'Walk',
            'bike': 'Bike',
            'train': 'Train',
            'subway': 'Subway',
            'railway': 'Train',
            'airplane': 'Airplane'
        }

    def process_all_users(self, max_users: int = None,
                          min_segment_length: int = 10) -> List[Dict]:
        """
        一次性处理所有用户数据

        返回:
            所有轨迹段的列表，每个元素包含完整信息
        """
        print("\n" + "=" * 80)
        print("GeoLife 基础数据预处理（所有实验通用）")
        print("=" * 80)

        users = self._get_all_users()
        if max_users:
            users = users[:max_users]

        print(f"\n找到 {len(users)} 个用户")

        all_segments = []

        for user_id in tqdm(users, desc="[处理用户]"):
            # 加载标签
            labels = self._load_labels(user_id)
            if labels.empty:
                continue

            # 处理该用户的所有轨迹
            trajectory_dir = os.path.join(
                self.geolife_root, f"Data/{user_id}/Trajectory"
            )

            if not os.path.exists(trajectory_dir):
                continue

            for traj_file in os.listdir(trajectory_dir):
                if not traj_file.endswith('.plt'):
                    continue

                traj_path = os.path.join(trajectory_dir, traj_file)
                trajectory_id = traj_file.replace('.plt', '')

                try:
                    # 加载并计算特征
                    trajectory = self._load_and_compute_features(traj_path)

                    if trajectory.empty or len(trajectory) < min_segment_length:
                        continue

                    # 按标签分割轨迹
                    segments = self._segment_trajectory(
                        trajectory, labels, user_id, trajectory_id
                    )

                    all_segments.extend(segments)

                except Exception as e:
                    warnings.warn(f"处理失败 {traj_path}: {e}")
                    continue

        print(f"\n✅ 预处理完成，共 {len(all_segments)} 个轨迹段")

        # 统计信息
        self._print_statistics(all_segments)

        return all_segments

    def _get_all_users(self) -> List[str]:
        """获取所有用户ID"""
        data_path = os.path.join(self.geolife_root, "Data")
        users = []
        for item in os.listdir(data_path):
            item_path = os.path.join(data_path, item)
            if os.path.isdir(item_path) and item.isdigit():
                users.append(item)
        return sorted(users)

    def _load_labels(self, user_id: str) -> pd.DataFrame:
        """加载用户标签"""
        labels_path = os.path.join(
            self.geolife_root, f"Data/{user_id}/labels.txt"
        )

        if not os.path.exists(labels_path):
            return pd.DataFrame()

        df = pd.read_csv(labels_path, sep='\t')
        df['Start Time'] = pd.to_datetime(df['Start Time'])
        df['End Time'] = pd.to_datetime(df['End Time'])

        return df

    def _load_and_compute_features(self, file_path: str) -> pd.DataFrame:
        """
        加载轨迹文件并计算9维特征

        核心特征计算（所有实验共用）：
        1. latitude, longitude (原始)
        2. speed, acceleration (运动特征)
        3. bearing_change (方向变化)
        4. distance, time_diff (基础)
        5. total_distance, total_time (累积)
        """
        # 1. 读取文件（处理6/7列格式）
        try:
            df = pd.read_csv(file_path, skiprows=6, header=None)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

        num_cols = df.shape[1]

        # 2. 标准化列名
        if num_cols == 7:
            df.columns = [
                'latitude', 'longitude', 'reserved',
                'altitude', 'date_days', 'date', 'time'
            ]
        elif num_cols == 6:
            df.columns = [
                'latitude', 'longitude', 'altitude',
                'date_days', 'date', 'time'
            ]
            df.insert(2, 'reserved', 0)
        else:
            return pd.DataFrame()

        # 3. 合并日期时间
        df['datetime'] = pd.to_datetime(
            df['date'] + ' ' + df['time'],
            format='%Y-%m-%d %H:%M:%S'
        )
        df = df.sort_values('datetime').reset_index(drop=True)

        # 4. 清洗无效坐标
        valid_mask = (
                (df['latitude'] >= -90) & (df['latitude'] <= 90) &
                (df['longitude'] >= -180) & (df['longitude'] <= 180)
        )
        df = df[valid_mask].reset_index(drop=True)

        if len(df) < 2:
            return pd.DataFrame()

        # 5. 向量化计算9维特征
        df = self._compute_trajectory_features(df)

        # 6. 只保留需要的列
        keep_cols = [
            'datetime', 'latitude', 'longitude',
            'speed', 'acceleration', 'bearing_change',
            'distance', 'time_diff',
            'total_distance', 'total_time'
        ]

        return df[keep_cols]

    def _compute_trajectory_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """向量化计算轨迹特征"""

        # 1. 时间差
        df['time_diff'] = df['datetime'].diff().dt.total_seconds().fillna(0)

        # 2. 距离（Haversine公式 - 向量化）
        lat1 = df['latitude'].shift(1).fillna(df['latitude'].iloc[0])
        lon1 = df['longitude'].shift(1).fillna(df['longitude'].iloc[0])
        lat2 = df['latitude']
        lon2 = df['longitude']

        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))

        distances = 6371000 * c  # 地球半径（米）
        distances.iloc[0] = 0.0
        df['distance'] = distances

        # 3. 速度和加速度
        time_diff_safe = df['time_diff'].replace(0, 1e-6)
        df['speed'] = df['distance'] / time_diff_safe
        df['acceleration'] = df['speed'].diff() / time_diff_safe
        df['acceleration'] = df['acceleration'].fillna(0)

        # 4. 方向变化
        y = np.sin(dlon) * np.cos(lat2_rad)
        x = np.cos(lat1_rad) * np.sin(lat2_rad) - \
            np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)

        bearing = np.degrees(np.arctan2(y, x))
        bearing = (bearing + 360) % 360
        bearing.iloc[0] = 0.0

        df['bearing'] = bearing
        df['bearing_change'] = df['bearing'].diff().abs().fillna(0)
        df['bearing_change'] = np.where(
            df['bearing_change'] > 180,
            360 - df['bearing_change'],
            df['bearing_change']
        )

        # 5. 累积特征
        df['total_distance'] = df['distance'].cumsum()
        df['total_time'] = df['time_diff'].cumsum()

        return df

    def _segment_trajectory(self, trajectory: pd.DataFrame,
                            labels: pd.DataFrame,
                            user_id: str,
                            trajectory_id: str) -> List[Dict]:
        """按标签分割轨迹"""
        segments = []

        for _, label_row in labels.iterrows():
            start_time = label_row['Start Time']
            end_time = label_row['End Time']
            raw_mode = str(label_row['Transportation Mode']).lower().strip()

            # 标签归一化
            mode = self.label_mapping.get(raw_mode, raw_mode.capitalize())

            # 时间范围过滤
            mask = (
                    (trajectory['datetime'] >= start_time) &
                    (trajectory['datetime'] <= end_time)
            )
            segment = trajectory[mask].copy()

            if len(segment) < 10:  # 最小长度过滤
                continue

            # 构建段信息
            segment_info = {
                'user_id': user_id,
                'trajectory_id': trajectory_id,
                'segment_id': f"{user_id}_{trajectory_id}_{start_time.strftime('%Y%m%d%H%M%S')}",
                'label': mode,
                'start_time': start_time,
                'end_time': end_time,
                'length': len(segment),
                'raw_points': segment.reset_index(drop=True),  # 包含9维特征
                'datetime_series': segment['datetime'].reset_index(drop=True)
            }

            segments.append(segment_info)

        return segments

    def _print_statistics(self, segments: List[Dict]):
        """打印统计信息"""
        from collections import Counter

        print("\n" + "=" * 80)
        print("数据统计")
        print("=" * 80)

        # 标签分布
        labels = [seg['label'] for seg in segments]
        label_counts = Counter(labels)

        print(f"\n总轨迹段数: {len(segments)}")
        print(f"\n标签分布:")
        for label, count in sorted(label_counts.items()):
            print(f"  {label:15s}: {count:6d} ({count / len(segments) * 100:.2f}%)")

        # 长度统计
        lengths = [seg['length'] for seg in segments]
        print(f"\n轨迹段长度:")
        print(f"  最小: {min(lengths)}")
        print(f"  最大: {max(lengths)}")
        print(f"  平均: {np.mean(lengths):.1f}")
        print(f"  中位数: {np.median(lengths):.1f}")

    def save_to_cache(self, segments: List[Dict], cache_path: str):
        """保存到缓存"""
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        with open(cache_path, 'wb') as f:
            pickle.dump(segments, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"\n✅ 数据已保存到: {cache_path}")
        print(f"   文件大小: {os.path.getsize(cache_path) / 1024 / 1024:.2f} MB")

    @staticmethod
    def load_from_cache(cache_path: str) -> List[Dict]:
        """从缓存加载"""
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"缓存文件不存在: {cache_path}")

        with open(cache_path, 'rb') as f:
            segments = pickle.load(f)

        print(f"\n✅ 从缓存加载: {len(segments)} 个轨迹段")
        return segments


def main():
    """示例：生成基础数据"""
    import argparse

    parser = argparse.ArgumentParser(description='GeoLife基础数据预处理')
    parser.add_argument(
        '--geolife_root',
        type=str,
        default='../data/Geolife Trajectories 1.3',
        help='GeoLife数据根目录'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='../data/processed/base_segments.pkl',
        help='输出缓存路径'
    )
    parser.add_argument(
        '--max_users',
        type=int,
        default=None,
        help='最大用户数（测试用）'
    )

    args = parser.parse_args()

    # 创建预处理器
    preprocessor = BaseGeoLifePreprocessor(args.geolife_root)

    # 处理数据
    segments = preprocessor.process_all_users(max_users=args.max_users)

    # 保存缓存
    preprocessor.save_to_cache(segments, args.output)

    print("\n" + "=" * 80)
    print("✅ 基础数据预处理完成！")
    print("=" * 80)
    print("\n后续实验可直接使用此缓存，无需重复处理GeoLife数据")


if __name__ == '__main__':
    main()