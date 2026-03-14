"""
增强知识图谱构建模块 (Exp3/Exp4 - 稳定版)
基于 Exp2 的架构，扩展 KG 特征从 11 维到 15 维

新增特征:
- 地铁入口 (1维)
- 共享单车点 (1维)
- 出租车点 (1维)
- 速度限制 (1维)
- 公交/地铁线路 (1维)

增强: 全面的异常处理，确保无 NaN/Inf 输出
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
    """增强版交通知识图谱 (Exp4 - 稳定版)"""

    # KG 特征维度常量
    KG_FEATURE_DIM = 15

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

        # KDTree 索引
        self.road_kdtree = None
        self.road_coords = None
        self.road_types = None

        self.poi_kdtree = None
        self.poi_coords = None
        self.poi_types = None

        # 新增数据结构
        self.speed_limits = {}  # 道路速度限制
        self.bus_routes = set()  # 公交线路经过的道路ID
        self.subway_routes = set()  # 地铁线路经过的道路ID

        # 网格缓存
        self._grid_cache = {}
        self._cache_resolution = 0.001
        self._cache_hits = 0
        self._cache_misses = 0

        # 初始化状态标志
        self._is_built = False

        # 默认特征向量（用于异常情况）
        self._default_features = np.zeros(self.KG_FEATURE_DIM, dtype=np.float32)

    def build_from_osm(self, road_network: pd.DataFrame, pois: pd.DataFrame,
                       transit_routes: pd.DataFrame = None):
        """从OSM数据构建增强知识图谱"""
        self.road_network = road_network
        self.pois = pois
        self.transit_routes = transit_routes

        try:
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

            self._is_built = True

        except Exception as e:
            print(f" ⚠️ KG 构建过程出现异常: {e}")
            print(" -> 将使用降级模式（返回零向量）")
            self._is_built = False

    def _add_road_network(self):
        """添加道路网络到知识图谱"""
        if self.road_network is None or len(self.road_network) == 0:
            return

        for _, road in self.road_network.iterrows():
            try:
                road_id = road['id']
                road_type = road.get('highway') or road.get('railway', '')
                coordinates = road.get('coordinates', [])

                if not coordinates:
                    continue

                if road['geometry_type'] == 'LineString':
                    coords = coordinates
                elif road['geometry_type'] == 'Polygon':
                    coords = coordinates[0] if coordinates else []
                else:
                    continue

                if not coords:
                    continue

                for i, coord in enumerate(coords):
                    try:
                        node_id = f"{road_id}_node_{i}"
                        lon, lat = float(coord[0]), float(coord[1])

                        # 验证坐标有效性
                        if np.isnan(lon) or np.isnan(lat) or np.isinf(lon) or np.isinf(lat):
                            continue

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
                            try:
                                distance = geodesic(
                                    (coords[i - 1][1], coords[i - 1][0]),
                                    (lat, lon)
                                ).meters
                            except Exception:
                                distance = 0.0

                            self.graph.add_edge(
                                prev_node_id, node_id,
                                type='road_segment',
                                road_type=road_type,
                                distance=distance
                            )
                    except Exception:
                        continue
            except Exception:
                continue

    def _extract_speed_limits(self):
        """提取速度限制信息"""
        if self.road_network is None:
            return

        for _, road in self.road_network.iterrows():
            try:
                road_id = road['id']
                maxspeed = road.get('maxspeed', None)

                if maxspeed:
                    speed_value = self._parse_speed(maxspeed)
                    if speed_value is not None:
                        self.speed_limits[road_id] = speed_value
            except Exception:
                continue

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
        if self.pois is None or len(self.pois) == 0:
            return

        for _, poi in self.pois.iterrows():
            try:
                poi_id = poi['id']
                poi_type = poi['type']
                coordinates = poi.get('coordinates', [])

                if not coordinates:
                    continue

                try:
                    if isinstance(coordinates[0], (list, tuple)):
                        lon, lat = float(coordinates[0][0]), float(coordinates[0][1])
                    else:
                        lon, lat = float(coordinates[0]), float(coordinates[1])
                except (IndexError, TypeError, ValueError):
                    continue

                # 验证坐标有效性
                if np.isnan(lon) or np.isnan(lat) or np.isinf(lon) or np.isinf(lat):
                    continue

                self.graph.add_node(
                    poi_id,
                    type='poi',
                    poi_type=poi_type,
                    name=poi.get('name', ''),
                    latitude=lat,
                    longitude=lon
                )
            except Exception:
                continue

    def _add_transit_routes(self):
        """优化后的线路添加逻辑"""
        if self.transit_routes is None or len(self.transit_routes) == 0:
            return

        for _, route in self.transit_routes.iterrows():
            try:
                route_type = route.get('route')
                members = route.get('members', [])

                for member in members:
                    m_id = member.get('ref') if isinstance(member, dict) else None
                    if m_id:
                        if route_type == 'bus':
                            self.bus_routes.add(str(m_id))
                        elif route_type == 'subway':
                            self.subway_routes.add(str(m_id))
            except Exception:
                continue

    def _build_spatial_indices(self):
        """预构建 KDTree 空间索引 - 稳定版"""
        # 构建道路节点索引
        road_nodes_data = []
        for n, d in self.graph.nodes(data=True):
            try:
                if d.get('type') == 'road_node':
                    lat = float(d.get('latitude', 0))
                    lon = float(d.get('longitude', 0))
                    road_type = d.get('road_type', '')

                    # 验证有效性
                    if not (np.isnan(lat) or np.isnan(lon) or np.isinf(lat) or np.isinf(lon)):
                        road_nodes_data.append((lat, lon, road_type))
            except Exception:
                continue

        if road_nodes_data:
            try:
                self.road_coords = np.array([
                    (lat, lon) for lat, lon, _ in road_nodes_data
                ], dtype=np.float64)
                self.road_types = [
                    self.road_type_mapping.get(road_type, 'unknown')
                    for _, _, road_type in road_nodes_data
                ]
                self.road_kdtree = KDTree(self.road_coords)
                print(f"   -> 道路 KDTree: {len(self.road_coords)} 个节点")
            except Exception as e:
                print(f"   ⚠️ 道路 KDTree 构建失败: {e}")
                self.road_kdtree = None

        # 构建 POI 索引
        poi_nodes_data = []
        for n, d in self.graph.nodes(data=True):
            try:
                if d.get('type') == 'poi':
                    lat = float(d.get('latitude', 0))
                    lon = float(d.get('longitude', 0))
                    poi_type = d.get('poi_type', '')

                    if not (np.isnan(lat) or np.isnan(lon) or np.isinf(lat) or np.isinf(lon)):
                        poi_nodes_data.append((lat, lon, poi_type))
            except Exception:
                continue

        if poi_nodes_data:
            try:
                self.poi_coords = np.array([
                    (lat, lon) for lat, lon, _ in poi_nodes_data
                ], dtype=np.float64)
                self.poi_types = [poi_type for _, _, poi_type in poi_nodes_data]
                self.poi_kdtree = KDTree(self.poi_coords)
                print(f"   -> POI KDTree: {len(self.poi_coords)} 个节点")
            except Exception as e:
                print(f"   ⚠️ POI KDTree 构建失败: {e}")
                self.poi_kdtree = None

    def extract_kg_features(self, trajectory: np.ndarray) -> np.ndarray:
        """
        提取 15 维增强 KG 特征 - 稳定版

        输入: (N, 9) 轨迹特征
        输出: (N, 15) KG 特征

        特征组成:
        - 道路类型 (6维): walk, bike, car, bus, train, unknown
        - 附近POI (6维): 公交站, 地铁入口, 停车场, 共享单车, 出租车, 最近POI距离
        - 道路属性 (2维): 速度限制, 是否在公交/地铁线路上
        - 道路密度 (1维): 附近道路节点数量

        保证：
        - 形状正确 (N, 15)
        - 无 NaN / Inf
        - dtype = float32
        """
        # 输入验证
        if trajectory is None or len(trajectory) == 0:
            return np.zeros((0, self.KG_FEATURE_DIM), dtype=np.float32)

        N = trajectory.shape[0]

        # 检查 KG 是否可用
        if not self._is_built or self.road_kdtree is None:
            return np.zeros((N, self.KG_FEATURE_DIM), dtype=np.float32)

        try:
            kg_features = np.zeros((N, self.KG_FEATURE_DIM), dtype=np.float32)

            uncached_indices = []
            uncached_coords = []

            # 步骤1: 检查缓存
            for i in range(N):
                try:
                    lat = float(trajectory[i, 0])
                    lon = float(trajectory[i, 1])

                    # 验证坐标有效性
                    if np.isnan(lat) or np.isnan(lon) or np.isinf(lat) or np.isinf(lon):
                        kg_features[i] = self._default_features
                        continue

                    grid_key = self._get_grid_key(lat, lon)

                    if grid_key in self._grid_cache:
                        kg_features[i] = self._grid_cache[grid_key]
                        self._cache_hits += 1
                    else:
                        uncached_indices.append(i)
                        uncached_coords.append([lat, lon])
                        self._cache_misses += 1
                except Exception:
                    kg_features[i] = self._default_features

            # 步骤2: 批量查询未缓存的点
            if uncached_indices:
                try:
                    uncached_coords = np.array(uncached_coords, dtype=np.float64)
                    uncached_features = self._batch_query_all_safe(uncached_coords)

                    # 步骤3: 更新缓存和结果
                    for i, idx in enumerate(uncached_indices):
                        try:
                            lat = float(trajectory[idx, 0])
                            lon = float(trajectory[idx, 1])
                            grid_key = self._get_grid_key(lat, lon)
                            self._grid_cache[grid_key] = uncached_features[i]
                            kg_features[idx] = uncached_features[i]
                        except Exception:
                            kg_features[idx] = self._default_features
                except Exception:
                    # 批量查询失败，使用默认值
                    for idx in uncached_indices:
                        kg_features[idx] = self._default_features

            # 最终安全检查
            kg_features = np.nan_to_num(kg_features, nan=0.0, posinf=0.0, neginf=0.0)
            kg_features = np.clip(kg_features, -10, 10)

            return kg_features.astype(np.float32)

        except Exception as e:
            # 完全失败，返回零向量
            return np.zeros((N, self.KG_FEATURE_DIM), dtype=np.float32)

    def _get_grid_key(self, lat: float, lon: float) -> Tuple[int, int]:
        """将坐标映射到网格"""
        return (
            round(lat / self._cache_resolution),
            round(lon / self._cache_resolution)
        )

    def _batch_query_all_safe(self, coords: np.ndarray) -> np.ndarray:
        """批量查询所有 15 维 KG 特征 - 安全版"""
        N = coords.shape[0]

        try:
            # 特征1: 道路类型 (6维)
            road_type_features = self._batch_query_road_types_safe(coords)

            # 特征2: 附近 POI (6维)
            poi_features = self._batch_query_pois_safe(coords)

            # 特征3: 道路属性 (2维)
            road_attr_features = self._batch_query_road_attributes_safe(coords)

            # 特征4: 道路密度 (1维)
            road_density = self._batch_query_road_density_safe(coords)

            result = np.concatenate([
                road_type_features,  # 6维
                poi_features,        # 6维
                road_attr_features,  # 2维
                road_density         # 1维
            ], axis=1)

            return result

        except Exception:
            return np.zeros((N, self.KG_FEATURE_DIM), dtype=np.float32)

    def _batch_query_road_types_safe(self, coords: np.ndarray,
                                      max_distance: float = 50.0) -> np.ndarray:
        """批量查询道路类型 (6维 one-hot) - 安全版"""
        N = coords.shape[0]
        road_type_features = np.zeros((N, 6), dtype=np.float32)

        if self.road_kdtree is None or self.road_coords is None:
            return road_type_features

        try:
            distances, indices = self.road_kdtree.query(coords, k=1)
            distances = distances * 111300.0  # 度转换为米

            type_names = ['walk', 'bike', 'car', 'bus', 'train', 'unknown']

            for i in range(N):
                try:
                    if distances[i] < max_distance and indices[i] < len(self.road_types):
                        road_type = self.road_types[indices[i]]
                        if road_type in type_names:
                            idx = type_names.index(road_type)
                            road_type_features[i, idx] = 1.0
                        else:
                            road_type_features[i, 5] = 1.0  # unknown
                    else:
                        road_type_features[i, 5] = 1.0  # unknown
                except Exception:
                    road_type_features[i, 5] = 1.0

        except Exception:
            pass

        return road_type_features

    def _batch_query_pois_safe(self, coords: np.ndarray,
                                max_distance: float = 200.0) -> np.ndarray:
        """批量查询 POI 信息 (6维) - 安全版"""
        N = coords.shape[0]
        poi_features = np.zeros((N, 6), dtype=np.float32)

        if self.poi_kdtree is None or self.poi_coords is None:
            return poi_features

        try:
            max_degree = max_distance / 111300.0
            indices = self.poi_kdtree.query_ball_point(coords, r=max_degree)

            for i in range(N):
                try:
                    if len(indices[i]) > 0:
                        nearby_types = [self.poi_types[j] for j in indices[i]
                                       if j < len(self.poi_types)]

                        # 公交站
                        if 'bus_stop' in nearby_types:
                            poi_features[i, 0] = 1.0

                        # 地铁入口
                        if 'subway_entrance' in nearby_types or 'station' in nearby_types:
                            poi_features[i, 1] = 1.0

                        # 停车场
                        if 'parking' in nearby_types:
                            poi_features[i, 2] = 1.0

                        # 共享单车点
                        if 'bicycle_rental' in nearby_types:
                            poi_features[i, 3] = 1.0

                        # 出租车停靠点
                        if 'taxi' in nearby_types:
                            poi_features[i, 4] = 1.0

                        # 最近POI距离 (归一化)
                        try:
                            poi_coords_nearby = self.poi_coords[indices[i]]
                            dists = np.linalg.norm(
                                poi_coords_nearby.astype(np.float64) - coords[i:i + 1].astype(np.float64),
                                axis=1
                            ) * 111300.0
                            poi_features[i, 5] = min(float(np.min(dists)) / 200.0, 1.0)
                        except Exception:
                            poi_features[i, 5] = 1.0
                except Exception:
                    continue

        except Exception:
            pass

        return poi_features

    def _batch_query_road_attributes_safe(self, coords: np.ndarray,
                                           max_distance: float = 50.0) -> np.ndarray:
        """批量查询道路属性 (2维) - 安全版"""
        N = coords.shape[0]
        road_attr = np.zeros((N, 2), dtype=np.float32)

        if self.road_kdtree is None:
            return road_attr

        try:
            distances, indices = self.road_kdtree.query(coords, k=1)
            distances = distances * 111300.0

            node_keys = list(self.graph.nodes())

            for i in range(N):
                try:
                    if distances[i] < max_distance and indices[i] < len(node_keys):
                        nearest_node_id = node_keys[indices[i]]
                        node_data = self.graph.nodes.get(nearest_node_id, {})
                        road_id = str(node_data.get('road_id', ''))

                        # 特征 1: 速度限制
                        speed = self.speed_limits.get(road_id, 60.0)
                        road_attr[i, 0] = min(float(speed) / 120.0, 1.0)

                        # 特征 2: 线路匹配
                        if road_id in self.bus_routes or road_id in self.subway_routes:
                            road_attr[i, 1] = 1.0
                except Exception:
                    continue

        except Exception:
            pass

        return road_attr

    def _batch_query_road_density_safe(self, coords: np.ndarray,
                                        radius: float = 100.0) -> np.ndarray:
        """批量查询道路密度 (1维) - 安全版"""
        N = coords.shape[0]
        densities = np.zeros((N, 1), dtype=np.float32)

        if self.road_kdtree is None:
            return densities

        try:
            radius_degree = radius / 111300.0
            indices = self.road_kdtree.query_ball_point(coords, r=radius_degree)

            for i in range(N):
                try:
                    count = len(indices[i]) if i < len(indices) else 0
                    densities[i, 0] = min(float(count) / 50.0, 1.0)
                except Exception:
                    continue

        except Exception:
            pass

        return densities

    def _link_roads_to_pois(self, max_distance: float = 100.0):
        """使用 KDTree 加速 POI 到最近道路节点的链接"""
        poi_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'poi']
        road_nodes_data = [(n, d) for n, d in self.graph.nodes(data=True) if d.get('type') == 'road_node']

        if not road_nodes_data or not poi_nodes:
            return

        try:
            print(f" -> 正在关联 {len(poi_nodes)} 个POI到 {len(road_nodes_data)} 个道路节点...")

            road_coords = np.array([[d['latitude'], d['longitude']] for n, d in road_nodes_data])
            road_node_ids = [n for n, d in road_nodes_data]
            road_tree = KDTree(road_coords)

            poi_coords_list = []
            poi_ids_list = []
            for poi_node in poi_nodes:
                try:
                    poi_data = self.graph.nodes[poi_node]
                    poi_coords_list.append([poi_data['latitude'], poi_data['longitude']])
                    poi_ids_list.append(poi_node)
                except Exception:
                    continue

            if not poi_coords_list:
                return

            poi_coords = np.array(poi_coords_list)
            max_degree_distance = max_distance / 111300.0
            indices = road_tree.query_ball_point(poi_coords, r=max_degree_distance)

            link_count = 0
            for i, neighbors_indices in enumerate(tqdm(indices, desc="   [KG关联进度]", leave=False)):
                try:
                    poi_node = poi_ids_list[i]
                    poi_lat, poi_lon = poi_coords[i]

                    min_dist = float('inf')
                    nearest_road = None

                    for j in neighbors_indices:
                        try:
                            road_node = road_node_ids[j]
                            road_lat, road_lon = road_coords[j]
                            dist = geodesic((poi_lat, poi_lon), (road_lat, road_lon)).meters

                            if dist < min_dist:
                                min_dist = dist
                                nearest_road = road_node
                        except Exception:
                            continue

                    if nearest_road is not None:
                        self.graph.add_edge(poi_node, nearest_road,
                                            type='nearby_road',
                                            distance=min_dist)
                        self.graph.add_edge(nearest_road, poi_node,
                                            type='has_poi',
                                            distance=min_dist)
                        link_count += 1
                except Exception:
                    continue

            print(f" -> 知识图谱道路-POI关联完成。共添加 {link_count} 条关联边。")
        except Exception as e:
            print(f" ⚠️ POI-道路关联失败: {e}")

    def get_cache_stats(self) -> Dict:
        """获取缓存统计"""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0

        return {
            'cache_size': len(self._grid_cache),
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': f"{hit_rate:.2%}",
            'cache_memory_mb': len(self._grid_cache) * self.KG_FEATURE_DIM * 4 / (1024 * 1024),
            'is_built': self._is_built
        }

    def save_cache(self, cache_path: str):
        """保存缓存到文件"""
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(self._grid_cache, f)
            print(f" -> 网格缓存已保存到: {cache_path}")
        except Exception as e:
            print(f" ⚠️ 缓存保存失败: {e}")

    def load_cache(self, cache_path: str):
        """从文件加载缓存"""
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    self._grid_cache = pickle.load(f)
                print(f" -> 网格缓存已加载: {len(self._grid_cache)} 个网格")
            except Exception as e:
                print(f" ⚠️ 缓存加载失败: {e}")

    def get_graph_statistics(self) -> Dict:
        """获取知识图谱统计信息"""
        try:
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
                'subway_routes': len(self.subway_routes),
                'is_built': self._is_built
            }
        except Exception:
            stats = {'is_built': self._is_built}
        return stats