"""
增强知识图谱构建模块 (Exp3)
基于 Exp2 的架构，扩展 KG 特征从 11 维到 15 维

新增特征:
- 地铁入口 (1维)
- 共享单车点 (1维)
- 出租车点 (1维)
- 速度限制 (1维)
- 公交/地铁线路 (1维)
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from geopy.distance import geodesic
import networkx as nx
from tqdm import tqdm
from scipy.spatial import KDTree
import re
import pickle
import os


class EnhancedTransportationKG:
    """增强版交通知识图谱 (Exp3)"""

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.road_network = None
        self.pois = None
        self.transit_routes = None

        # 道路类型映射
        self.road_type_mapping = {
            'footway': 'walk',
            'cycleway': 'bike',
            'primary': 'car',
            'secondary': 'car',
            'tertiary': 'car',
            'residential': 'car',
            'motorway': 'car',
            'trunk': 'car',
            'bus_stop': 'bus',
            'station': 'train',
            'subway_entrance': 'train',
            'parking': 'car'
        }

        # KDTree 索引 (继承自 Exp2)
        self.road_kdtree = None
        self.road_coords = None
        self.road_types = None

        self.poi_kdtree = None
        self.poi_coords = None
        self.poi_types = None

        # Exp3 新增数据结构
        self.speed_limits = {}  # 道路速度限制
        self.bus_routes = set()  # 公交线路经过的道路ID
        self.subway_routes = set()  # 地铁线路经过的道路ID

        # 网格缓存 (继承自 Exp2)
        self._grid_cache = {}
        self._cache_resolution = 0.001
        self._cache_hits = 0
        self._cache_misses = 0

    def build_from_osm(self, road_network: pd.DataFrame, pois: pd.DataFrame,
                       transit_routes: pd.DataFrame = None):
        """从OSM数据构建增强知识图谱"""
        self.road_network = road_network
        self.pois = pois
        self.transit_routes = transit_routes

        print("\n -> 正在添加道路节点和路段...")
        self._add_road_network()
        print(f" -> 道路网络添加完成。当前图节点数: {self.graph.number_of_nodes()}")

        print(" -> 正在提取速度限制信息...")
        self._extract_speed_limits()
        print(f" -> 提取到 {len(self.speed_limits)} 条速度限制记录")

        print(" -> 正在添加 POI 节点...")
        self._add_pois()
        print(f" -> POI 节点添加完成。当前图节点数: {self.graph.number_of_nodes()}")

        if transit_routes is not None and len(transit_routes) > 0:
            print(" -> 正在添加公交/地铁线路...")
            self._add_transit_routes()
            print(f" -> 公交线路: {len(self.bus_routes)}, 地铁线路: {len(self.subway_routes)}")

        print(" -> 正在构建 KDTree 空间索引...")
        self._build_spatial_indices()
        print(" -> KDTree 索引构建完成！")

        self._link_roads_to_pois()

    def _add_road_network(self):
        """添加道路网络到知识图谱"""
        for _, road in self.road_network.iterrows():
            road_id = road['id']
            road_type = road.get('highway') or road.get('railway', '')
            coordinates = road.get('coordinates', [])

            if not coordinates:
                continue

            if road['geometry_type'] == 'LineString':
                coords = coordinates
            elif road['geometry_type'] == 'Polygon':
                coords = coordinates[0]
            else:
                continue

            for i, coord in enumerate(coords):
                node_id = f"{road_id}_node_{i}"
                lon, lat = float(coord[0]), float(coord[1])

                self.graph.add_node(
                    node_id,
                    type='road_node',
                    road_id=road_id,
                    road_type=road_type,
                    latitude=lat,
                    longitude=lon
                )

                if i > 0:
                    prev_node_id = f"{road_id}_node_{i - 1}"
                    distance = geodesic(
                        (coords[i - 1][1], coords[i - 1][0]),
                        (lat, lon)
                    ).meters

                    self.graph.add_edge(
                        prev_node_id, node_id,
                        type='road_segment',
                        road_type=road_type,
                        distance=distance
                    )

    def _extract_speed_limits(self):
        """提取速度限制信息"""
        for _, road in self.road_network.iterrows():
            road_id = road['id']
            maxspeed = road.get('maxspeed', None)

            if maxspeed:
                speed_value = self._parse_speed(maxspeed)
                if speed_value is not None:
                    self.speed_limits[road_id] = speed_value

    def _parse_speed(self, speed_str) -> Optional[float]:
        """解析速度字符串"""
        if not speed_str:
            return None

        try:
            speed_str = str(speed_str).strip().lower()
            numbers = re.findall(r'\d+', speed_str)
            if not numbers:
                return None

            speed = float(numbers[0])

            if 'mph' in speed_str:
                speed = speed * 1.60934

            return speed

        except (ValueError, AttributeError):
            return None

    def _add_pois(self):
        """添加POI到知识图谱"""
        for _, poi in self.pois.iterrows():
            poi_id = poi['id']
            poi_type = poi['type']
            coordinates = poi.get('coordinates', [])

            if not coordinates:
                continue

            if isinstance(coordinates[0], list):
                lon, lat = float(coordinates[0][0]), float(coordinates[0][1])
            else:
                lon, lat = float(coordinates[0]), float(coordinates[1])

            self.graph.add_node(
                poi_id,
                type='poi',
                poi_type=poi_type,
                name=poi.get('name', ''),
                latitude=lat,
                longitude=lon
            )

    def _add_transit_routes(self):
        """优化后的线路添加逻辑"""
        if self.transit_routes is None:
            return

        for _, route in self.transit_routes.iterrows():
            route_type = route.get('route')
            members = route.get('members', [])

            for member in members:
                # 这里的 ref 需要与 road_network 中的 id 格式对齐 (例如 'way/123')
                m_id = member.get('ref')
                if m_id:
                    if route_type == 'bus':
                        self.bus_routes.add(str(m_id))
                    elif route_type == 'subway':
                        self.subway_routes.add(str(m_id))

    def _batch_query_road_attributes(self, coords: np.ndarray, max_distance: float = 50.0) -> np.ndarray:
        N = coords.shape[0]
        road_attr = np.zeros((N, 2), dtype=np.float32)

        distances, indices = self.road_kdtree.query(coords, k=1)
        distances = distances * 111300.0

        # 获取图中所有节点的 key，用于快速索引
        node_keys = list(self.graph.nodes())

        for i in range(N):
            if distances[i] < max_distance:
                nearest_node_id = node_keys[indices[i]]
                node_data = self.graph.nodes[nearest_node_id]
                road_id = str(node_data.get('road_id', ''))

                # 特征 1: 速度
                speed = self.speed_limits.get(road_id, 60.0)
                road_attr[i, 0] = min(speed / 120.0, 1.0)

                # 特征 2: 线路匹配 (核心修复)
                if road_id in self.bus_routes or road_id in self.subway_routes:
                    road_attr[i, 1] = 1.0

        return road_attr

    def _build_spatial_indices(self):
        """预构建 KDTree 空间索引 (继承自 Exp2)"""
        # 构建道路节点索引
        road_nodes_data = [
            (d['latitude'], d['longitude'], d['road_type'])
            for n, d in self.graph.nodes(data=True)
            if d.get('type') == 'road_node'
        ]

        if road_nodes_data:
            self.road_coords = np.array([
                (float(lat), float(lon)) for lat, lon, _ in road_nodes_data
            ], dtype=np.float64)
            self.road_types = [
                self.road_type_mapping.get(road_type, 'unknown')
                for _, _, road_type in road_nodes_data
            ]
            self.road_kdtree = KDTree(self.road_coords)
            print(f"   -> 道路 KDTree: {len(self.road_coords)} 个节点")

        # 构建 POI 索引
        poi_nodes_data = [
            (d['latitude'], d['longitude'], d['poi_type'])
            for n, d in self.graph.nodes(data=True)
            if d.get('type') == 'poi'
        ]

        if poi_nodes_data:
            self.poi_coords = np.array([
                (float(lat), float(lon)) for lat, lon, _ in poi_nodes_data
            ], dtype=np.float64)
            self.poi_types = [poi_type for _, _, poi_type in poi_nodes_data]
            self.poi_kdtree = KDTree(self.poi_coords)
            print(f"   -> POI KDTree: {len(self.poi_coords)} 个节点")

    def extract_kg_features(self, trajectory: np.ndarray) -> np.ndarray:
        """
        提取 15 维增强 KG 特征 (混合缓存 + 批量查询)

        输入: (N, 9) 轨迹特征
        输出: (N, 15) KG 特征

        特征组成:
        - 道路类型 (6维): walk, bike, car, bus, train, unknown
        - 附近POI (6维): 公交站, 地铁入口, 停车场, 共享单车, 出租车, 最近POI距离
        - 道路属性 (2维): 速度限制, 是否在公交/地铁线路上
        - 道路密度 (1维): 附近道路节点数量
        """
        if self.road_kdtree is None or self.poi_kdtree is None:
            return np.zeros((trajectory.shape[0], 15), dtype=np.float32)

        N = trajectory.shape[0]
        kg_features = np.zeros((N, 15), dtype=np.float32)

        uncached_indices = []
        uncached_coords = []

        # 步骤1: 检查缓存
        for i in range(N):
            lat, lon = float(trajectory[i, 0]), float(trajectory[i, 1])
            grid_key = self._get_grid_key(lat, lon)

            if grid_key in self._grid_cache:
                kg_features[i] = self._grid_cache[grid_key]
                self._cache_hits += 1
            else:
                uncached_indices.append(i)
                uncached_coords.append([lat, lon])
                self._cache_misses += 1

        # 步骤2: 批量查询未缓存的点
        if uncached_indices:
            uncached_coords = np.array(uncached_coords, dtype=np.float64)
            uncached_features = self._batch_query_all(uncached_coords)

            # 步骤3: 更新缓存和结果
            for i, idx in enumerate(uncached_indices):
                lat, lon = float(trajectory[idx, 0]), float(trajectory[idx, 1])
                grid_key = self._get_grid_key(lat, lon)
                self._grid_cache[grid_key] = uncached_features[i]
                kg_features[idx] = uncached_features[i]

        return kg_features.astype(np.float32)

    def _get_grid_key(self, lat: float, lon: float) -> Tuple[int, int]:
        """将坐标映射到网格"""
        return (
            round(lat / self._cache_resolution),
            round(lon / self._cache_resolution)
        )

    def _batch_query_all(self, coords: np.ndarray) -> np.ndarray:
        """批量查询所有 15 维 KG 特征"""
        # 特征1: 道路类型 (6维)
        road_type_features = self._batch_query_road_types(coords)

        # 特征2: 附近 POI (6维) - 增强版
        poi_features = self._batch_query_pois_enhanced(coords)

        # 特征3: 道路属性 (2维) - 新增
        road_attr_features = self._batch_query_road_attributes(coords)

        # 特征4: 道路密度 (1维)
        road_density = self._batch_query_road_density(coords)

        return np.concatenate([
            road_type_features,  # 6维
            poi_features,  # 6维
            road_attr_features,  # 2维
            road_density  # 1维
        ], axis=1)

    def _batch_query_road_types(self, coords: np.ndarray,
                                 max_distance: float = 50.0) -> np.ndarray:
        """批量查询道路类型 (6维 one-hot)"""
        N = coords.shape[0]

        distances, indices = self.road_kdtree.query(coords, k=1)
        distances = distances * 111300.0

        type_names = ['walk', 'bike', 'car', 'bus', 'train', 'unknown']
        road_type_features = np.zeros((N, 6), dtype=np.float32)

        for i in range(N):
            if distances[i] < max_distance:
                road_type = self.road_types[indices[i]]
                if road_type in type_names:
                    idx = type_names.index(road_type)
                    road_type_features[i, idx] = 1.0
            else:
                road_type_features[i, 5] = 1.0  # unknown

        return road_type_features

    def _batch_query_pois_enhanced(self, coords: np.ndarray,
                                    max_distance: float = 200.0) -> np.ndarray:
        """
        批量查询 POI 信息 (6维 - 增强版)
        [公交站, 地铁入口, 停车场, 共享单车, 出租车, 最近POI距离]
        """
        N = coords.shape[0]
        poi_features = np.zeros((N, 6), dtype=np.float32)

        max_degree = max_distance / 111300.0
        indices = self.poi_kdtree.query_ball_point(coords, r=max_degree)

        for i in range(N):
            if len(indices[i]) > 0:
                nearby_types = [self.poi_types[j] for j in indices[i]]

                # 公交站
                if 'bus_stop' in nearby_types:
                    poi_features[i, 0] = 1.0

                # 地铁入口 (新增)
                if 'subway_entrance' in nearby_types or 'station' in nearby_types:
                    poi_features[i, 1] = 1.0

                # 停车场
                if 'parking' in nearby_types:
                    poi_features[i, 2] = 1.0

                # 共享单车点 (新增)
                if 'bicycle_rental' in nearby_types:
                    poi_features[i, 3] = 1.0

                # 出租车停靠点 (新增)
                if 'taxi' in nearby_types:
                    poi_features[i, 4] = 1.0

                # 最近POI距离 (归一化)
                poi_coords_nearby = self.poi_coords[indices[i]]
                dists = np.linalg.norm(
                    poi_coords_nearby.astype(np.float64) - coords[i:i + 1].astype(np.float64),
                    axis=1
                ) * 111300.0
                poi_features[i, 5] = min(dists) / 200.0

        return poi_features

    def _batch_query_road_density(self, coords: np.ndarray,
                                   radius: float = 100.0) -> np.ndarray:
        """批量查询道路密度 (1维)"""
        N = coords.shape[0]

        radius_degree = radius / 111300.0
        indices = self.road_kdtree.query_ball_point(coords, r=radius_degree)

        densities = np.array([len(idx) for idx in indices], dtype=np.float32)
        densities = np.clip(densities / 50.0, 0, 1)

        return densities.reshape(-1, 1)

    def _link_roads_to_pois(self, max_distance: float = 100.0):
        """使用 KDTree 加速 POI 到最近道路节点的链接"""
        poi_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'poi']
        road_nodes_data = [(n, d) for n, d in self.graph.nodes(data=True) if d.get('type') == 'road_node']

        if not road_nodes_data:
            print("警告: 无道路节点，跳过关联。")
            return

        print(f" -> 正在关联 {len(poi_nodes)} 个POI到 {len(road_nodes_data)} 个道路节点...")

        road_coords = np.array([[d['latitude'], d['longitude']] for n, d in road_nodes_data])
        road_node_ids = [n for n, d in road_nodes_data]
        road_tree = KDTree(road_coords)

        poi_coords_list = []
        poi_ids_list = []
        for poi_node in poi_nodes:
            poi_data = self.graph.nodes[poi_node]
            poi_coords_list.append([poi_data['latitude'], poi_data['longitude']])
            poi_ids_list.append(poi_node)

        poi_coords = np.array(poi_coords_list)
        max_degree_distance = max_distance / 111300.0
        indices = road_tree.query_ball_point(poi_coords, r=max_degree_distance)

        link_count = 0
        for i, neighbors_indices in enumerate(tqdm(indices, desc="   [KG关联进度]", leave=False)):
            poi_node = poi_ids_list[i]
            poi_lat, poi_lon = poi_coords[i]

            min_dist = float('inf')
            nearest_road = None

            for j in neighbors_indices:
                road_node = road_node_ids[j]
                road_lat, road_lon = road_coords[j]
                dist = geodesic((poi_lat, poi_lon), (road_lat, road_lon)).meters

                if dist < min_dist:
                    min_dist = dist
                    nearest_road = road_node

            if nearest_road is not None:
                self.graph.add_edge(poi_node, nearest_road,
                                    type='nearby_road',
                                    distance=min_dist)
                self.graph.add_edge(nearest_road, poi_node,
                                    type='has_poi',
                                    distance=min_dist)
                link_count += 1

        print(f" -> 知识图谱道路-POI关联完成。共添加 {link_count} 条关联边。")

    def get_cache_stats(self) -> Dict:
        """获取缓存统计"""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0

        return {
            'cache_size': len(self._grid_cache),
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': f"{hit_rate:.2%}",
            'cache_memory_mb': len(self._grid_cache) * 15 * 4 / (1024 * 1024)
        }

    def save_cache(self, cache_path: str):
        """保存缓存到文件"""
        with open(cache_path, 'wb') as f:
            pickle.dump(self._grid_cache, f)
        print(f" -> 网格缓存已保存到: {cache_path}")

    def load_cache(self, cache_path: str):
        """从文件加载缓存"""
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                self._grid_cache = pickle.load(f)
            print(f" -> 网格缓存已加载: {len(self._grid_cache)} 个网格")

    def get_graph_statistics(self) -> Dict:
        """获取知识图谱统计信息"""
        stats = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'road_nodes': len([n for n, d in self.graph.nodes(data=True)
                               if d.get('type') == 'road_node']),
            'poi_nodes': len([n for n, d in self.graph.nodes(data=True)
                              if d.get('type') == 'poi']),
            'poi_links': len([u for u, v, k, d in self.graph.edges(data=True, keys=True)
                              if d.get('type') == 'nearby_road']),
            'speed_limits': len(self.speed_limits),
            'bus_routes': len(self.bus_routes),
            'subway_routes': len(self.subway_routes)
        }
        return stats