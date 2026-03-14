"""
数据预处理模块 (Exp3)
继承 Exp2 的所有优化，新增公交/地铁线路提取
"""
import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import json
import warnings
from tqdm import tqdm

pd.options.mode.chained_assignment = None


class GeoLifeDataLoader:
    """GeoLife数据加载器 (完全继承自 Exp2)"""

    def __init__(self, data_root: str):
        self.data_root = data_root

    def load_trajectory(self, file_path: str) -> pd.DataFrame:
        """加载单个轨迹文件，鲁棒地处理 6 列或 7 列数据"""
        try:
            df = pd.read_csv(file_path, skiprows=6, header=None)
        except pd.errors.EmptyDataError:
            warnings.warn(f"文件 {file_path} 为空，跳过。")
            return pd.DataFrame()

        num_cols = df.shape[1]

        # 根据列数分配列名
        if num_cols == 7:
            df.columns = ['latitude', 'longitude', 'reserved', 'altitude', 'date_days', 'date', 'time']
            df['reserved'] = df['reserved'].replace({0.0: 0, 0: 0}).astype(int)
        elif num_cols == 6:
            try:
                df.columns = ['latitude', 'longitude', 'altitude', 'date_days', 'date', 'time']
                df.insert(2, 'reserved', 0)
            except ValueError:
                df.columns = ['latitude', 'longitude', 'reserved', 'date_days', 'date', 'time']
                df.insert(3, 'altitude', np.nan)
                df['reserved'] = df['reserved'].replace({0.0: 0, 0: 0}).astype(int)
            warnings.warn(f"文件 {os.path.basename(file_path)} 只有 6 列，已标准化为 7 列。")
        else:
            raise ValueError(f"文件 {file_path} 列数为 {num_cols}，无法处理。")

        # 合并日期时间
        # 速度快：告诉 Pandas 格式，跳过“猜测”阶段
        df['datetime'] = pd.to_datetime(
            df['date'] + ' ' + df['time'],
            format='%Y-%m-%d %H:%M:%S'
        )
        df = df.sort_values('datetime').reset_index(drop=True)

        # 数据清洗：删除无效坐标
        invalid_lat_mask = (df['latitude'] < -90) | (df['latitude'] > 90)
        invalid_lon_mask = (df['longitude'] < -180) | (df['longitude'] > 180)

        if invalid_lat_mask.any() or invalid_lon_mask.any():
            warnings.warn(f"文件 {os.path.basename(file_path)} 发现无效坐标，正在删除。")
            df = df[~invalid_lat_mask & ~invalid_lon_mask].reset_index(drop=True)
            if len(df) < 2:
                return pd.DataFrame()

        # 向量化计算特征
        df = self._calculate_features_vectorized(df)

        # 清理不必要的列
        df = df.drop(columns=['date', 'time', 'date_days'], errors='ignore')

        return df

    def _calculate_features_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """向量化计算 9 维轨迹特征 (完全继承自 Exp2)"""
        # 1. 时间差
        df['time_diff'] = df['datetime'].diff().dt.total_seconds().fillna(0)

        # 2. 距离 (Haversine)
        lat1 = df['latitude'].shift(1).fillna(df['latitude'].iloc[0])
        lon1 = df['longitude'].shift(1).fillna(df['longitude'].iloc[0])
        lat2 = df['latitude']
        lon2 = df['longitude']

        lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(np.radians, [lat1, lon1, lat2, lon2])

        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad

        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
        c = 2 * np.arcsin(np.sqrt(a))

        R = 6371000
        distances = R * c
        distances.iloc[0] = 0.0
        df['distance'] = distances

        # 3. 速度和加速度
        time_diff_safe = df['time_diff'].replace(0, 1e-6)
        df['speed'] = df['distance'] / time_diff_safe
        df['acceleration'] = df['speed'].diff() / time_diff_safe
        df['acceleration'] = df['acceleration'].fillna(0)

        # 4. 方向
        df['bearing'] = self._calculate_bearing_vectorized(
            lat1_rad.values, lon1_rad.values, lat2_rad.values, lon2_rad.values
        )

        # 5. 方向变化
        df['bearing_change'] = df['bearing'].diff().abs().fillna(0)
        df['bearing_change'] = np.where(
            df['bearing_change'] > 180,
            360 - df['bearing_change'],
            df['bearing_change']
        )

        # 6. 累积特征
        df['total_distance'] = df['distance'].cumsum()
        df['total_time'] = df['time_diff'].cumsum()

        return df

    def _calculate_bearing_vectorized(self, lat1, lon1, lat2, lon2):
        """向量化计算方位角"""
        dlon = lon2 - lon1
        y = np.sin(dlon) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        bearing = np.degrees(np.arctan2(y, x))
        bearing = (bearing + 360) % 360
        bearing[0] = 0.0
        return bearing

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
        segments = []
        # --- 核心修改：类别映射逻辑 ---
        label_mapping = {
            'taxi': 'Car & taxi', 'car': 'Car & taxi',
            'bus': 'Bus', 'walk': 'Walk', 'bike': 'Bike',
            'train': 'Train', 'subway': 'Subway', 'airplane': 'Airplane'
        }

        for _, row in labels.iterrows():
            st, et = pd.to_datetime(row['Start Time']), pd.to_datetime(row['End Time'])
            raw_mode = str(row['Transportation Mode']).lower().strip()
            mode = label_mapping.get(raw_mode, raw_mode.capitalize())  # 未定义的则首字母大写

            mask = (trajectory['datetime'] >= st) & (trajectory['datetime'] <= et)
            seg = trajectory[mask].copy()
            if len(seg) >= 10:
                segments.append((seg, mode))
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
    """OSM数据加载器 (Exp3 - 新增 transit_routes 提取)"""

    def __init__(self, geojson_path: str):
        self.geojson_path = geojson_path

    def load_osm_data(self) -> Dict:
        """加载OSM GeoJSON数据（支持大文件流式加载）"""
        file_size = os.path.getsize(self.geojson_path) / (1024 * 1024)
        print(f"加载OSM数据文件: {self.geojson_path} (大小: {file_size:.2f} MB)")

        if file_size > 100:
            print("检测到大文件，使用流式加载...")
            return self._load_osm_data_streaming()
        else:
            with open(self.geojson_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data

    def _load_osm_data_streaming(self) -> Dict:
        """流式加载大文件"""
        try:
            import ijson

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
            print("警告: ijson未安装，使用标准JSON加载")
            print("建议安装: pip install ijson")
            with open(self.geojson_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"加载完成，共 {len(data.get('features', []))} 个特征")
            return data

    def extract_road_network(self, osm_data: Dict) -> pd.DataFrame:
        """提取道路网络信息（包含速度限制）"""
        roads = []

        for feature in osm_data.get('features', []):
            props = feature.get('properties', {})
            geometry = feature.get('geometry', {})

            highway = props.get('highway', '')
            railway = props.get('railway', '')
            maxspeed = props.get('maxspeed', None)  # 新增：速度限制

            if highway or railway:
                road_info = {
                    'id': props.get('@id', '') or props.get('id', ''),
                    'highway': highway,
                    'railway': railway,
                    'maxspeed': maxspeed,
                    'geometry_type': geometry.get('type', ''),
                    'coordinates': geometry.get('coordinates', [])
                }
                roads.append(road_info)

        print(f"提取到 {len(roads)} 条道路")
        return pd.DataFrame(roads)

    def extract_pois(self, osm_data: Dict) -> pd.DataFrame:
        """提取POI信息（包含新增的地铁入口、共享单车、出租车点）"""
        pois = []

        # Exp3 扩展的 POI 类型
        poi_types = [
            'bus_stop',
            'station',
            'subway_entrance',  # 新增
            'parking',
            'taxi',  # 新增
            'bicycle_rental'  # 新增
        ]

        for feature in osm_data.get('features', []):
            props = feature.get('properties', {})
            geometry = feature.get('geometry', {})

            highway = props.get('highway', '')
            railway = props.get('railway', '')
            amenity = props.get('amenity', '')
            public_transport = props.get('public_transport', '')

            # 检查是否匹配 POI 类型
            poi_type = None
            if highway in poi_types:
                poi_type = highway
            elif railway in poi_types:
                poi_type = railway
            elif amenity in poi_types:
                poi_type = amenity
            elif public_transport == 'station':
                poi_type = 'station'

            if poi_type:
                poi_info = {
                    'id': props.get('@id', '') or props.get('id', ''),
                    'type': poi_type,
                    'name': props.get('name', ''),
                    'coordinates': geometry.get('coordinates', [])
                }
                pois.append(poi_info)

        print(f"提取到 {len(pois)} 个POI")
        return pd.DataFrame(pois)

    # data_preprocessing.py 内部修改

    def extract_transit_routes(self, osm_data: Dict) -> pd.DataFrame:
        """
        提取公交和地铁线路信息 (Exp3 修复版)
        支持从 properties 直接获取或从 @relations 嵌套结构中提取
        """
        routes = []

        for feature in osm_data.get('features', []):
            props = feature.get('properties', {})

            # 策略 1: 直接在 properties 中 (标准 Relation 导出)
            route_type = props.get('route', '')

            # 策略 2: 嵌套在 @relations 中 (如你提供的样本)
            relations = props.get('@relations', [])

            # 整合所有可能的 route 信息
            found_routes = []
            if route_type in ['bus', 'subway']:
                found_routes.append({
                    'route': route_type,
                    'ref': props.get('ref', '')
                })

            for rel in relations:
                reltags = rel.get('reltags', {})
                if reltags.get('route') in ['bus', 'subway']:
                    found_routes.append({
                        'route': reltags.get('route'),
                        'ref': reltags.get('ref', '')
                    })

            # 如果找到线路，记录其成员 (道路ID)
            if found_routes:
                # 获取成员，通常在 properties 的 @members 或 feature 的 geometry 关联中
                # 这里简化处理：将该 feature 本身的 ID 视为线路的一部分
                members = props.get('@members', []) or [{"ref": props.get('@id')}]

                for r in found_routes:
                    routes.append({
                        'id': props.get('@id', ''),
                        'route': r['route'],
                        'ref': r['ref'],
                        'members': members
                    })

        print(f"提取到 {len(routes)} 条公交/地铁线路信息")
        return pd.DataFrame(routes)


def preprocess_trajectory_segments(segments: List[Tuple[pd.DataFrame, str]],
                                   min_length: int = 10) -> List[Tuple[np.ndarray, str]]:
    """
    预处理轨迹段，提取 9 维特征 (完全继承自 Exp2)
    """
    processed_segments = []
    FIXED_SEQUENCE_LENGTH = 50

    feature_cols = [
        'latitude', 'longitude', 'speed', 'acceleration',
        'bearing_change', 'distance', 'time_diff',
        'total_distance', 'total_time'
    ]

    for segment, label in tqdm(segments, desc="[轨迹段预处理]"):
        if len(segment) < min_length:
            continue

        features = segment[feature_cols].values
        current_length = len(features)

        # 序列长度规范化
        if current_length > FIXED_SEQUENCE_LENGTH:
            indices = np.linspace(0, current_length - 1, FIXED_SEQUENCE_LENGTH, dtype=int)
            features = features[indices]
        elif current_length < FIXED_SEQUENCE_LENGTH:
            padding = np.zeros((FIXED_SEQUENCE_LENGTH - current_length, features.shape[1]))
            features = np.vstack([features, padding])

        processed_segments.append((features, label))

    return processed_segments