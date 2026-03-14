"""
GeoLife数据加载和预处理模块 (Robust and Enhanced Version)
处理GeoLife轨迹数据，并确保数据加载的鲁棒性。

关键增强点:
1. 鲁棒地处理 6 列或 7 列 GeoLife 文件。
2. 强制清洗无效的经纬度坐标点。
3. 轨迹特征（包括9维）计算已完全向量化，提高性能。
4. 修复：新增 get_all_users 方法。

注意：已移除序列独立归一化，改为在 train_maso.py 中进行全局归一化。
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Tuple
from geopy.distance import geodesic
import warnings
from tqdm import tqdm

# 暂时忽略设置副本警告
pd.options.mode.chained_assignment = None

class GeoLifeDataLoader:
    """GeoLife数据加载器"""

    def __init__(self, data_root: str):
        self.data_root = data_root

    def get_all_users(self) -> List[str]:
        """扫描数据根目录，获取所有用户ID"""
        data_dir = os.path.join(self.data_root, 'Data')
        if not os.path.isdir(data_dir):
            # 兼容 GeoLife Trajectories 1.3 结构
            data_dir = self.data_root

        if not os.path.isdir(data_dir):
             # 再次检查
             print(f"警告: GeoLife数据目录未找到: {self.data_root}")
             return []

        # 筛选出看起来像用户ID的目录（GeoLife用户ID通常是三位数字）
        users = [d for d in os.listdir(data_dir)
                 if os.path.isdir(os.path.join(data_dir, d)) and len(d) == 3 and d.isdigit()]

        return sorted(users)

    def load_trajectory(self, file_path: str) -> pd.DataFrame:
        """
        加载单个轨迹文件，鲁棒地处理 6 列或 7 列数据，并进行数据清洗。
        """

        # 1. 读取原始数据，跳过前6行元数据
        try:
            df = pd.read_csv(file_path, skiprows=6, header=None)
        except pd.errors.EmptyDataError:
            warnings.warn(f"文件 {file_path} 为空，跳过。")
            return pd.DataFrame()

        num_cols = df.shape[1]

        # 2. 根据列数分配正确的列名并进行标准化 (增强鲁棒性)
        if num_cols == 7:
            # 标准 GeoLife 格式：7列
            df.columns = ['latitude', 'longitude', 'reserved', 'altitude', 'date_days', 'date', 'time']
            df['reserved'] = df['reserved'].replace({0.0: 0, 0: 0}).astype(int)
        elif num_cols == 6:
            # 非标准 GeoLife 格式：6列。这里采用 GeoLife 常见的缺失 'reserved' 列的假设。
            try:
                df.columns = ['latitude', 'longitude', 'altitude', 'date_days', 'date', 'time']
                df.insert(2, 'reserved', 0) # 插入 reserved 填充 0
            except ValueError:
                # 备选：如果缺失 Altitude
                df.columns = ['latitude', 'longitude', 'reserved', 'date_days', 'date', 'time']
                df.insert(3, 'altitude', np.nan)
                df['reserved'] = df['reserved'].replace({0.0: 0, 0: 0}).astype(int)

            warnings.warn(f"文件 {os.path.basename(file_path)} 只有 6 列，已尝试标准化为 7 列。")
        else:
            # 无法处理的列数
            return pd.DataFrame()


        # 删除 GeoLife 的 'reserved'/'unused' 列 (Field 3, 几乎总是 0)
        df = df.drop('reserved', axis=1, errors='ignore')

        # 3. 合并日期时间
        # 针对您日志中的 UserWarning: Could not infer format，这里不进行修正，保留原样以减少非核心修改
        # 速度快：告诉 Pandas 格式，跳过“猜测”阶段
        df['datetime'] = pd.to_datetime(
            df['date'] + ' ' + df['time'],
            format='%Y-%m-%d %H:%M:%S'
        )
        df = df.sort_values('datetime').reset_index(drop=True)

        # 4. 强制数据清洗：删除无效坐标点 (增强健壮性)
        invalid_lat_mask = (df['latitude'] < -90) | (df['latitude'] > 90)
        invalid_lon_mask = (df['longitude'] < -180) | (df['longitude'] > 180)

        if invalid_lat_mask.any() or invalid_lon_mask.any():
            warnings.warn(f"文件 {os.path.basename(file_path)} 发现无效坐标，正在删除。")
            df = df[~invalid_lat_mask & ~invalid_lon_mask].reset_index(drop=True)

            if len(df) < 2:
                # 轨迹太短，无法计算特征，直接返回空DataFrame
                return pd.DataFrame()

        # 5. 向量化计算特征
        df = self._calculate_features(df)

        # 清理不必要的列
        df = df.drop(columns=['date', 'time', 'date_days', 'altitude'], errors='ignore')

        return df

    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """向量化计算轨迹特征，包括 9 维特征"""

        # 1. 计算时间差（秒）
        df['time_diff'] = df['datetime'].diff().dt.total_seconds().fillna(0)

        # 2. 向量化计算距离 (Haversine 公式)
        lat1 = df['latitude'].shift(1).fillna(df['latitude'].iloc[0])
        lon1 = df['longitude'].shift(1).fillna(df['longitude'].iloc[0])
        lat2 = df['latitude']
        lon2 = df['longitude']

        lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(np.radians, [lat1, lon1, lat2, lon2])

        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad

        a = np.sin(dlat/2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2.0)**2
        c = 2 * np.arcsin(np.sqrt(a))

        R = 6371000 # 地球半径，单位：米
        distances = R * c

        distances.iloc[0] = 0.0
        df['distance'] = distances

        # 3. 向量化计算速度和加速度
        # 避免除以 0，使用一个极小的数代替 0
        time_diff_safe = df['time_diff'].replace(0, 1e-6)

        df['speed'] = df['distance'] / time_diff_safe

        df['acceleration'] = df['speed'].diff() / time_diff_safe
        df['acceleration'] = df['acceleration'].fillna(0)

        # 4. 向量化计算方向（Bearing）
        df['bearing'] = self._calculate_bearing_vectorized(lat1_rad.values, lon1_rad.values, lat2_rad.values, lon2_rad.values)

        # 5. 计算方向变化率
        df['bearing_change'] = df['bearing'].diff().abs().fillna(0)
        # 处理角度跨越 360/0 度的变化 (取较小值)
        df['bearing_change'] = np.where(df['bearing_change'] > 180, 360 - df['bearing_change'], df['bearing_change'])

        # 6. 扩展特征：累计距离和总时长 (9维特征所需)
        df['total_distance'] = df['distance'].cumsum()
        df['total_time'] = df['time_diff'].cumsum()

        return df

    def _calculate_bearing_vectorized(self, lat1: np.ndarray, lon1: np.ndarray,
                                        lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
        """向量化计算方位角（度）"""

        dlon = lon2 - lon1

        y = np.sin(dlon) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)

        bearing = np.degrees(np.arctan2(y, x))

        bearing = (bearing + 360) % 360

        bearing[0] = 0.0

        return bearing

    def load_labels(self, user_id: str) -> pd.DataFrame:
        """加载用户标签数据"""
        # 适应 GeoLife Trajectories 1.3 的结构
        labels_path = os.path.join(self.data_root, f"{user_id}/labels.txt")
        if not os.path.exists(labels_path):
            # 尝试 GeoLife Trajectories 1.3/Data/用户ID/labels.txt 结构
            labels_path = os.path.join(self.data_root, f"Data/{user_id}/labels.txt")

        if not os.path.exists(labels_path):
            return pd.DataFrame()

        # 原始 GeoLife labels.txt 使用 Tab 分隔符
        df = pd.read_csv(labels_path, sep='\t')
        df['Start Time'] = pd.to_datetime(df['Start Time'])
        df['End Time'] = pd.to_datetime(df['End Time'])
        return df

    def segment_trajectory(self, trajectory: pd.DataFrame, labels: pd.DataFrame) -> List[Tuple[pd.DataFrame, str]]:
        """根据标签分割轨迹"""
        segments = []

        if labels.empty:
            # 如果没有标签，将整个轨迹作为一个未知段返回（用于预测）
            return [(trajectory, 'unknown')]

        for _, label_row in labels.iterrows():
            start_time = label_row['Start Time']
            end_time = label_row['End Time']
            mode = label_row['Transportation Mode']

            mask = (trajectory['datetime'] >= start_time) & (trajectory['datetime'] <= end_time)
            segment = trajectory[mask].copy()

            if len(segment) > 0:
                segments.append((segment, mode))

        return segments


def preprocess_segments(segments: List[Tuple[pd.DataFrame, str]],
                       min_length: int = 10,
                       max_length: int = 200,
                       target_length: int = 100) -> List[Tuple[np.ndarray, str]]:
    """
    预处理轨迹段，转换为固定长度的序列，并进行标签重映射。
    确保所有输出序列的长度都严格等于 target_length。
    注意：此函数不再进行归一化，归一化操作在 train_maso.py 中使用全局统计量完成。
    """
    processed_segments = []

    # =================================================================
    # 🔥 核心修改：标签重映射逻辑定义 (统一 'taxi' 和 'car' 的标签)
    # =================================================================
    # 定义映射规则：将 'taxi' 归类到 'car'，统一首字母大写
    MAPPING = {
        'taxi': 'Car',
    }
    # 定义合并后的最终标签名称
    NEW_CLASS_NAME = 'Car & taxi'
    # =================================================================


    # 9 维特征列表：
    feature_cols = ['latitude', 'longitude', 'speed', 'acceleration',
                    'bearing_change', 'distance', 'time_diff',
                    'total_distance', 'total_time']

    # 使用 tqdm 封装以显示进度条
    for segment, label in tqdm(segments, desc="[轨迹段预处理]"):
        if len(segment) < min_length:
            continue

        # 1. 提取特征序列（9维特征）
        features = segment[feature_cols].values
        L = len(features)

        # 2. 序列长度处理 (确保最终长度为 target_length)
        if L >= target_length:
            # 序列长度大于等于 target_length，需要裁剪/采样

            if L > max_length:
                # 序列太长 (> max_length)，进行均匀采样
                indices = np.linspace(0, L-1, target_length, dtype=int)
                features = features[indices]
            elif L > target_length:
                # 序列在 [target_length, max_length] 之间，随机裁剪到 target_length
                # L - target_length + 1 是可以作为起始点的最大索引 (保证裁剪到 target_length)
                start_index = np.random.randint(0, L - target_length + 1)
                features = features[start_index:start_index + target_length]
            else: # L == target_length
                pass # 长度刚好，不操作

        elif L < target_length:
            # 如果序列太短 (< target_length)，进行零填充
            padding = np.zeros((target_length - L, features.shape[1]))
            features = np.vstack([features, padding])


        # 3. 🔥 核心修改：应用标签重映射和重命名，统一首字母大写
        mapped_label = MAPPING.get(label, label)

        if mapped_label == 'Car':
            final_label = NEW_CLASS_NAME
        else:
            # 统一首字母大写
            final_label = mapped_label.capitalize() if mapped_label else mapped_label

        processed_segments.append((features, final_label)) # 使用最终的标签

    print(f"\n标签重映射完成: 'taxi' 已并入 '{NEW_CLASS_NAME}'")
    return processed_segments