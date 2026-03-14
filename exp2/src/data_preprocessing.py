"""
数据预处理模块 (Optimized and Robust Version)
处理GeoLife轨迹数据和OSM数据

关键优化点:
1. GeoLife DataLoader 能够鲁棒地处理 6 列和 7 列格式。
2. 实现了经纬度异常值的强制清洗，避免数据错误中断。
3. 轨迹特征计算（距离、方位角、累计量）已完全向量化，大幅提升了性能。
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict
import json
import warnings
from tqdm import tqdm

# 暂时忽略设置副本警告
pd.options.mode.chained_assignment = None

class GeoLifeDataLoader:
    """GeoLife数据加载器"""

    def __init__(self, data_root: str):
        self.data_root = data_root

    def load_trajectory(self, file_path: str) -> pd.DataFrame:
        """加载单个轨迹文件，鲁棒地处理 6 列或 7 列数据，并进行数据清洗"""

        # 1. 读取原始数据，跳过前6行元数据
        try:
            df = pd.read_csv(file_path, skiprows=6, header=None)
        except pd.errors.EmptyDataError:
            warnings.warn(f"文件 {file_path} 为空，跳过。")
            return pd.DataFrame()

        num_cols = df.shape[1]

        # 2. 根据列数分配正确的列名并进行标准化
        if num_cols == 7:
            # 标准 GeoLife 格式：7列
            df.columns = ['latitude', 'longitude', 'reserved', 'altitude', 'date_days', 'date', 'time']
            df['reserved'] = df['reserved'].replace({0.0: 0, 0: 0}).astype(int)
        elif num_cols == 6:
            # 非标准 GeoLife 格式：6列。我们假设缺失 Reserved 列。
            try:
                # 假设缺失 Reserved (第3列)
                df.columns = ['latitude', 'longitude', 'altitude', 'date_days', 'date', 'time']
                df.insert(2, 'reserved', 0) # 插入 reserved 填充 0
            except ValueError:
                # 假设缺失 Altitude (第4列)
                df.columns = ['latitude', 'longitude', 'reserved', 'date_days', 'date', 'time']
                df.insert(3, 'altitude', np.nan)  # 插入 altitude 填充 NaN
                df['reserved'] = df['reserved'].replace({0.0: 0, 0: 0}).astype(int)

            warnings.warn(f"文件 {os.path.basename(file_path)} 只有 6 列，已尝试标准化为 7 列。")
        else:
            raise ValueError(f"文件 {file_path} 列数为 {num_cols}，既非 6 也非 7。无法处理。")

        # 3. 合并日期时间
        df['datetime'] = pd.to_datetime(
            df['date'] + ' ' + df['time'],
            format='%Y-%m-%d %H:%M:%S'
        )
        df = df.sort_values('datetime').reset_index(drop=True)

        # 4. 强制数据清洗：删除无效坐标点
        invalid_lat_mask = (df['latitude'] < -90) | (df['latitude'] > 90)
        invalid_lon_mask = (df['longitude'] < -180) | (df['longitude'] > 180)

        if invalid_lat_mask.any() or invalid_lon_mask.any():
            warnings.warn(f"文件 {os.path.basename(file_path)} 发现无效坐标，正在删除。")
            df = df[~invalid_lat_mask & ~invalid_lon_mask].reset_index(drop=True)

            if len(df) < 2:
                raise ValueError("轨迹文件在清洗后点数过少。")

        # 5. 向量化计算特征
        df = self._calculate_features_vectorized(df)

        # 清理不必要的列
        df = df.drop(columns=['date', 'time', 'date_days'])

        return df

    # -----------------------------------------------------------
    # V E C T O R I Z E D   F E A T U R E S (高性能计算)
    # -----------------------------------------------------------

    def _calculate_features_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算轨迹特征 (使用向量化优化性能)，包括 9 维所需的所有特征"""

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
        time_diff_safe = df['time_diff'].replace(0, 1e-6) # 避免除以 0

        df['speed'] = df['distance'] / time_diff_safe

        df['acceleration'] = df['speed'].diff() / time_diff_safe
        df['acceleration'] = df['acceleration'].fillna(0)

        # 4. 向量化计算方向（Bearing）
        df['bearing'] = self._calculate_bearing_vectorized(lat1_rad.values, lon1_rad.values, lat2_rad.values, lon2_rad.values)

        # 5. 计算方向变化率
        df['bearing_change'] = df['bearing'].diff().abs().fillna(0)

        # 处理角度跨越 360/0 度的变化 (取较小值)
        df['bearing_change'] = np.where(df['bearing_change'] > 180, 360 - df['bearing_change'], df['bearing_change'])

        # -----------------------------------------------------------
        # 6. 扩展特征：累计距离和总时长 (NEW: 9维特征所需)
        # -----------------------------------------------------------

        # total_distance: 轨迹段的总累计距离 (从第一个点开始累加)
        df['total_distance'] = df['distance'].cumsum()

        # total_time: 轨迹段的总累计时长 (从第一个点开始累加)
        df['total_time'] = df['time_diff'].cumsum()

        return df

    def _calculate_bearing_vectorized(self, lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
        """向量化计算方位角（度）"""

        dlon = lon2 - lon1

        y = np.sin(dlon) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)

        bearing = np.degrees(np.arctan2(y, x))

        bearing = (bearing + 360) % 360

        bearing[0] = 0.0

        return bearing

    # -----------------------------------------------------------
    # O T H E R   M E T H O D S (标签归一化已添加)
    # -----------------------------------------------------------

    def _normalize_mode(self, mode: str) -> str:
        """
        根据 7 大类标准合并交通方式标签（任务定义统一）。
        7 大类: Walk, Bike, Bus, Car & taxi, Train, Subway, Airplane
        """
        mode_lower = mode.lower().strip()

        # 类别合并
        if mode_lower in ['car', 'taxi', 'drive']:
            return 'Car & taxi'
        elif mode_lower in ['train', 'railway', 'high-speed-rail']:
            return 'Train'
        elif mode_lower == 'subway':
            return 'Subway'
        elif mode_lower == 'walk':
            return 'Walk'
        elif mode_lower == 'bike':
            return 'Bike'
        elif mode_lower == 'bus':
            return 'Bus'
        elif mode_lower == 'airplane':
            return 'Airplane'
        else:
            # 不在7类中的标签，返回None，后续过滤
            return None


    def load_labels(self, user_id: str) -> pd.DataFrame:
        """加载用户标签数据"""
        labels_path = os.path.join(self.data_root, f"Data/{user_id}/labels.txt")
        if not os.path.exists(labels_path):
            return pd.DataFrame()

        df = pd.read_csv(labels_path, sep='\t')
        df['Start Time'] = pd.to_datetime(df['Start Time'])
        df['End Time'] = pd.to_datetime(df['End Time'])
        return df

    def segment_trajectory(self, trajectory: pd.DataFrame, labels: pd.DataFrame) -> List[Tuple[pd.DataFrame, str]]:
        """根据标签分割轨迹，并进行标签归一化/合并为 7 大类（任务定义统一）"""
        segments = []

        for _, label_row in labels.iterrows():
            start_time = label_row['Start Time']
            end_time = label_row['End Time']
            mode = label_row['Transportation Mode']

            # 标签归一化/合并步骤
            normalized_mode = self._normalize_mode(mode)

            # 过滤不在7类中的标签
            if normalized_mode is None:
                continue

            mask = (trajectory['datetime'] >= start_time) & (trajectory['datetime'] <= end_time)
            segment = trajectory[mask].copy()

            if len(segment) > 0:
                segments.append((segment, normalized_mode))

        return segments

    def get_all_users(self) -> List[str]:
        """获取所有用户ID"""
        data_path = os.path.join(self.data_root, "Data")
        users = []
        for item in os.listdir(data_path):
            item_path = os.path.join(data_path, item)
            if os.path.isdir(item_path) and item.isdigit():
                users.append(item)
        return sorted(users)


class OSMDataLoader:
    """OSM数据加载器"""

    def __init__(self, geojson_path: str):
        self.geojson_path = geojson_path

    def load_osm_data(self) -> Dict:
        """加载OSM GeoJSON数据（支持大文件）"""
        import os

        file_size = os.path.getsize(self.geojson_path) / (1024 * 1024)  # MB
        print(f"加载OSM数据文件: {self.geojson_path} (大小: {file_size:.2f} MB)")

        # 对于大文件，使用流式加载
        if file_size > 100:  # 大于100MB
            print("检测到大文件，使用流式加载...")
            return self._load_osm_data_streaming()
        else:
            with open(self.geojson_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data

    def _load_osm_data_streaming(self) -> Dict:
        """流式加载大文件（逐行解析）"""
        try:
            import ijson  # 需要安装: pip install ijson

            with open(self.geojson_path, 'rb') as f:
                parser = ijson.items(f, 'features.item')
                features = []
                count = 0

                for feature in parser:
                    features.append(feature)
                    count += 1
                    if count % 10000 == 0:
                        print(f"  已加载 {count} 个特征...")

                print(f"总共加载 {count} 个特征")

                return {
                    'type': 'FeatureCollection',
                    'features': features
                }
        except ImportError:
            print("警告: ijson未安装，使用标准JSON加载（可能较慢且占用内存）")
            print("建议安装: pip install ijson 以获得更好的大文件处理性能")
            print("正在加载...")
            with open(self.geojson_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"加载完成，共 {len(data.get('features', []))} 个特征")
            return data
        except Exception as e:
            print(f"流式加载失败: {e}")
            print("回退到标准JSON加载...")
            with open(self.geojson_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data

    def extract_road_network(self, osm_data: Dict) -> pd.DataFrame:
        """提取道路网络信息"""
        roads = []

        for feature in osm_data.get('features', []):
            props = feature.get('properties', {})
            geometry = feature.get('geometry', {})

            highway = props.get('highway', '')
            railway = props.get('railway', '')

            if highway or railway:
                road_info = {
                    'id': props.get('@id', '') or props.get('id', ''),
                    'highway': highway,
                    'railway': railway,
                    'geometry_type': geometry.get('type', ''),
                    'coordinates': geometry.get('coordinates', [])
                }
                roads.append(road_info)

        print(f"提取到 {len(roads)} 条道路")
        return pd.DataFrame(roads)

    def extract_pois(self, osm_data: Dict) -> pd.DataFrame:
        """提取POI信息（公交站、地铁站等）"""
        pois = []

        for feature in osm_data.get('features', []):
            props = feature.get('properties', {})
            geometry = feature.get('geometry', {})

            highway = props.get('highway', '')
            railway = props.get('railway', '')
            amenity = props.get('amenity', '')

            poi_types = ['bus_stop', 'station', 'parking', 'taxi', 'subway_entrance']

            if (highway in poi_types or
                railway in poi_types or
                amenity in poi_types or
                highway == 'bus_stop' or
                railway == 'station' or
                amenity in ['parking', 'taxi']):

                poi_type = highway or railway or amenity
                poi_info = {
                    'id': props.get('@id', '') or props.get('id', ''),
                    'type': poi_type,
                    'name': props.get('name', ''),
                    'coordinates': geometry.get('coordinates', [])
                }
                pois.append(poi_info)

        print(f"提取到 {len(pois)} 个POI")
        return pd.DataFrame(pois)


# src/data_preprocessing.py 文件中的函数
def preprocess_trajectory_segments(segments: List[Tuple[pd.DataFrame, str]],
                                   min_length: int = 10) -> List[Tuple[np.ndarray, str]]:
    """预处理轨迹段，转换为固定长度的序列，提取 9 维特征"""
    processed_segments = []

    # 设定固定的目标序列长度 (所有张量都将是 [50, 9] 的形状)
    FIXED_SEQUENCE_LENGTH = 50

    # 9 维特征列表：
    feature_cols = ['latitude', 'longitude', 'speed', 'acceleration',
                    'bearing_change', 'distance', 'time_diff',
                    'total_distance', 'total_time']

    for segment, label in tqdm(segments, desc="[轨迹段预处理]"):
        if len(segment) < min_length:
            continue

        features = segment[feature_cols].values
        current_length = len(features)

        # 1. 轨迹段长度规范化
        if current_length > FIXED_SEQUENCE_LENGTH:
            # 采样：如果太长（>50），采样到固定长度 50
            indices = np.linspace(0, current_length - 1, FIXED_SEQUENCE_LENGTH, dtype=int)
            features = features[indices]

        elif current_length < FIXED_SEQUENCE_LENGTH:
            # 填充：如果太短（<50），用零填充到固定长度 50
            padding = np.zeros((FIXED_SEQUENCE_LENGTH - current_length, features.shape[1]))
            features = np.vstack([features, padding])

        processed_segments.append((features, label))

    return processed_segments