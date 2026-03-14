"""
知识图谱构建模块 (混合优化版 - 修复 Decimal 类型问题)
结合网格缓存和批量查询，实现速度与准确率的最佳平衡

核心优化:
1. ✅ 预构建 KDTree 索引（一次性初始化）
2. ✅ 网格缓存系统（常用位置缓存）
3. ✅ 批量查询（未缓存点）
4. ✅ 自适应缓存策略
5. ✅ 修复 Decimal 类型问题
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from geopy.distance import geodesic
import networkx as nx
from collections import defaultdict
from tqdm import tqdm
from scipy.spatial import KDTree
import pickle
import os
from decimal import Decimal


class TransportationKnowledgeGraph:
    """交通知识图谱 (混合优化版)"""

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.road_network = None
        self.pois = None
        self.road_type_mapping = {
            'footway': 'walk',
            'cycleway': 'bike',
            'primary': 'car',
            'secondary': 'car',
            'tertiary': 'car',
            'residential': 'car',
            'bus_stop': 'bus',
            'station': 'train',
            'parking': 'car'
        }

        # KDTree 索引
        self.road_kdtree = None
        self.road_coords = None
        self.road_types = None

        self.poi_kdtree = None
        self.poi_coords = None
        self.poi_types = None

        # 网格缓存
        self._grid_cache = {}
        self._cache_resolution = 0.001  # 约 111 米
        self._cache_hits = 0
        self._cache_misses = 0

    def build_from_osm(self, road_network: pd.DataFrame, pois: pd.DataFrame):
        """从OSM数据构建知识图谱"""
        self.road_network = road_network
        self.pois = pois

        print(" -> 正在添加道路节点和路段...")
        self._add_road_network()
        print(f" -> 道路网络添加完成。当前图节点数: {self.graph.number_of_nodes()}")

        print(" -> 正在添加 POI 节点...")
        self._add_pois()
        print(f" -> POI 节点添加完成。当前图节点数: {self.graph.number_of_nodes()}")

        print(" -> 正在构建 KDTree 空间索引...")
        self._build_spatial_indices()
        print(" -> KDTree 索引构建完成！")

        self._link_roads_to_pois()

    def _convert_to_float(self, value):
        """转换任意数值类型为 float"""
        if isinstance(value, Decimal):
            return float(value)
        return float(value)

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
                # 修复：转换 Decimal 为 float
                lon, lat = self._convert_to_float(coord[0]), self._convert_to_float(coord[1])

                self.graph.add_node(node_id,
                                  type='road_node',
                                  road_id=road_id,
                                  road_type=road_type,
                                  latitude=lat,
                                  longitude=lon)

                if i > 0:
                    prev_node_id = f"{road_id}_node_{i-1}"
                    distance = geodesic((coords[i-1][1], coords[i-1][0]),
                                       (lat, lon)).meters

                    self.graph.add_edge(prev_node_id, node_id,
                                      type='road_segment',
                                      road_type=road_type,
                                      distance=distance)

    def _add_pois(self):
        """添加POI到知识图谱"""
        for _, poi in self.pois.iterrows():
            poi_id = poi['id']
            poi_type = poi['type']
            coordinates = poi.get('coordinates', [])

            if not coordinates:
                continue

            if isinstance(coordinates[0], list):
                # 修复：转换 Decimal 为 float
                lon, lat = self._convert_to_float(coordinates[0][0]), self._convert_to_float(coordinates[0][1])
            else:
                # 修复：转换 Decimal 为 float
                lon, lat = self._convert_to_float(coordinates[0]), self._convert_to_float(coordinates[1])

            self.graph.add_node(poi_id,
                              type='poi',
                              poi_type=poi_type,
                              name=poi.get('name', ''),
                              latitude=lat,
                              longitude=lon)

    def _build_spatial_indices(self):
        """预构建 KDTree 空间索引"""

        # 1. 构建道路节点索引
        road_nodes_data = [
            (d['latitude'], d['longitude'], d['road_type'])
            for n, d in self.graph.nodes(data=True)
            if d.get('type') == 'road_node'
        ]

        if road_nodes_data:
            # 确保坐标是 float 类型
            self.road_coords = np.array([
                (float(lat), float(lon)) for lat, lon, _ in road_nodes_data
            ], dtype=np.float64)
            self.road_types = [
                self.road_type_mapping.get(road_type, 'unknown')
                for _, _, road_type in road_nodes_data
            ]
            self.road_kdtree = KDTree(self.road_coords)
            print(f"   -> 道路 KDTree: {len(self.road_coords)} 个节点")

        # 2. 构建 POI 索引
        poi_nodes_data = [
            (d['latitude'], d['longitude'], d['poi_type'])
            for n, d in self.graph.nodes(data=True)
            if d.get('type') == 'poi'
        ]

        if poi_nodes_data:
            # 确保坐标是 float 类型
            self.poi_coords = np.array([
                (float(lat), float(lon)) for lat, lon, _ in poi_nodes_data
            ], dtype=np.float64)
            self.poi_types = [poi_type for _, _, poi_type in poi_nodes_data]
            self.poi_kdtree = KDTree(self.poi_coords)
            print(f"   -> POI KDTree: {len(self.poi_coords)} 个节点")

    # ========== 核心：混合查询策略 ==========
    def extract_kg_features(self, trajectory: np.ndarray) -> np.ndarray:
        """
        混合方案：网格缓存 + 批量查询

        Args:
            trajectory: (N, 9) 轨迹数组

        Returns:
            kg_features: (N, 11) KG特征数组
        """
        if self.road_kdtree is None or self.poi_kdtree is None:
            return np.zeros((trajectory.shape[0], 11), dtype=np.float32)

        N = trajectory.shape[0]
        kg_features = np.zeros((N, 11), dtype=np.float32)

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
        """将坐标映射到网格 (约 111 米精度)"""
        return (
            round(lat / self._cache_resolution),
            round(lon / self._cache_resolution)
        )

    def _batch_query_all(self, coords: np.ndarray) -> np.ndarray:
        """批量查询所有 KG 特征"""
        # 特征1: 道路类型 (6维)
        road_type_features = self._batch_query_road_types(coords)

        # 特征2: 附近 POI (4维)
        poi_features = self._batch_query_pois(coords)

        # 特征3: 道路密度 (1维)
        road_density = self._batch_query_road_density(coords)

        return np.concatenate([
            road_type_features,
            poi_features,
            road_density
        ], axis=1)

    def _batch_query_road_types(self, coords: np.ndarray,
                                 max_distance: float = 50.0) -> np.ndarray:
        """批量查询道路类型"""
        N = coords.shape[0]

        # 查询最近的道路节点
        distances, indices = self.road_kdtree.query(coords, k=1)
        distances = distances * 111300.0  # 转换为米

        # 构建 one-hot 编码
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

    def _batch_query_pois(self, coords: np.ndarray,
                          max_distance: float = 200.0) -> np.ndarray:
        """批量查询 POI 信息（修复 Decimal 类型问题）"""
        N = coords.shape[0]
        poi_features = np.zeros((N, 4), dtype=np.float32)

        # 查询附近所有 POI
        max_degree = max_distance / 111300.0
        indices = self.poi_kdtree.query_ball_point(coords, r=max_degree)

        for i in range(N):
            if len(indices[i]) > 0:
                # 获取附近 POI 类型
                nearby_types = [self.poi_types[j] for j in indices[i]]

                # 特征编码
                if 'bus_stop' in nearby_types:
                    poi_features[i, 0] = 1.0
                if 'station' in nearby_types:
                    poi_features[i, 1] = 1.0
                if 'parking' in nearby_types:
                    poi_features[i, 2] = 1.0

                # 最近 POI 距离（修复 Decimal 问题）
                poi_coords_nearby = self.poi_coords[indices[i]]
                # 确保 coords 和 poi_coords_nearby 都是 float64
                dists = np.linalg.norm(
                    poi_coords_nearby.astype(np.float64) - coords[i:i+1].astype(np.float64),
                    axis=1
                ) * 111300.0
                poi_features[i, 3] = min(dists) / 200.0  # 归一化

        return poi_features

    def _batch_query_road_density(self, coords: np.ndarray,
                                   radius: float = 100.0) -> np.ndarray:
        """批量查询道路密度"""
        N = coords.shape[0]

        # 查询附近道路节点数量
        radius_degree = radius / 111300.0
        indices = self.road_kdtree.query_ball_point(coords, r=radius_degree)

        # 计算密度并归一化
        densities = np.array([len(idx) for idx in indices], dtype=np.float32)
        densities = np.clip(densities / 50.0, 0, 1)

        return densities.reshape(-1, 1)

    # ========== 缓存管理 ==========
    def get_cache_stats(self) -> Dict:
        """获取缓存统计"""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0

        return {
            'cache_size': len(self._grid_cache),
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': f"{hit_rate:.2%}",
            'cache_memory_mb': len(self._grid_cache) * 11 * 4 / (1024 * 1024)
        }

    def save_cache(self, cache_path: str):
        """保存缓存到文件"""
        with open(cache_path, 'wb') as f:
            pickle.dump(self._grid_cache, f)
        print(f" -> 缓存已保存到: {cache_path}")

    def load_cache(self, cache_path: str):
        """从文件加载缓存"""
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                self._grid_cache = pickle.load(f)
            print(f" -> 缓存已加载: {len(self._grid_cache)} 个网格")
        else:
            print(f" -> 缓存文件不存在: {cache_path}")

    def clear_cache(self):
        """清空缓存"""
        self._grid_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    # ========== 保留原有接口 ==========
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

    def get_graph_statistics(self) -> Dict:
        """获取知识图谱统计信息"""
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'road_nodes': len([n for n, d in self.graph.nodes(data=True)
                             if d.get('type') == 'road_node']),
            'poi_nodes': len([n for n, d in self.graph.nodes(data=True)
                            if d.get('type') == 'poi']),
            'poi_links': len([u for u, v, k, d in self.graph.edges(data=True, keys=True)
                              if d.get('type') == 'nearby_road'])
        }