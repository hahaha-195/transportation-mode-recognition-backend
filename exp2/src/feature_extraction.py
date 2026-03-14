"""
特征提取模块 (混合优化版)
结合轨迹特征和知识图谱特征

关键点:
- 调用 kg.extract_kg_features() 进行批量特征提取
- 不包含任何循环或嵌套 tqdm
- 保持简洁高效
"""
import numpy as np
from typing import Tuple

from .knowledge_graph import TransportationKnowledgeGraph


class FeatureExtractor:
    """特征提取器 (混合优化版)"""

    def __init__(self, kg: TransportationKnowledgeGraph):
        """
        初始化特征提取器

        Args:
            kg: 交通知识图谱实例
        """
        self.kg = kg

    def extract_features(self, trajectory: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        提取轨迹特征和知识图谱特征

        关键：这里只调用一次 kg.extract_kg_features()，
              不会出现嵌套循环问题

        Args:
            trajectory: 已预处理的 NumPy 数组 (N, 9)
                [:, 0] = latitude
                [:, 1] = longitude
                [:, 2] = speed
                [:, 3] = acceleration
                [:, 4] = bearing_change
                [:, 5] = distance
                [:, 6] = time_diff
                [:, 7] = total_distance
                [:, 8] = total_time

        Returns:
            trajectory_features: 归一化后的轨迹特征 (N, 9)
            kg_features: 知识图谱特征 (N, 11)
        """
        # 1. 提取和归一化轨迹特征
        trajectory_features = self._extract_trajectory_features(trajectory)

        # 2. 提取知识图谱特征（关键：这里是批量提取，不是逐点循环）
        try:
            # 这里调用 kg.extract_kg_features(trajectory)
            # 在 knowledge_graph.py 中实现了：
            #   - 向量化网格键生成
            #   - 批量缓存查询
            #   - 批量 KDTree 查询
            # 因此不会出现嵌套循环
            kg_features = self.kg.extract_kg_features(trajectory)
        except Exception as e:
            # 如果 KG 特征提取失败，使用零填充
            print(f"警告: KG 特征提取失败 ({e}). 使用零填充代替。")
            kg_features = np.zeros((trajectory.shape[0], 11), dtype=np.float32)

        # 3. 验证维度
        if kg_features.shape[1] != 11:
            raise ValueError(f"KG 特征维度错误：预期 11 维，实际 {kg_features.shape[1]} 维。")

        if trajectory_features.shape[1] != 9:
            raise ValueError(f"轨迹特征维度错误：预期 9 维，实际 {trajectory_features.shape[1]} 维。")

        return trajectory_features, kg_features

    def _extract_trajectory_features(self, trajectory: np.ndarray) -> np.ndarray:
        """
        提取并归一化轨迹特征

        Args:
            trajectory: (N, 9) 原始轨迹特征

        Returns:
            normalized_features: (N, 9) 归一化后的特征
        """
        # 复制数组避免修改原始数据
        features = trajectory.copy()

        # Z-score 归一化
        features = self._normalize_features(features)

        return features

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        归一化特征 (Z-score 归一化)

        公式: z = (x - μ) / σ

        Args:
            features: (N, 9) 原始特征

        Returns:
            normalized: (N, 9) 归一化特征
        """
        # 计算均值和标准差（沿着第 0 维，即对每个特征维度单独计算）
        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True) + 1e-8  # 避免除以零

        # 归一化
        normalized = (features - mean) / std

        # 截断异常值到 [-5, 5] 范围
        normalized = np.clip(normalized, -5, 5)

        return normalized

    def combine_features(self, trajectory_features: np.ndarray,
                         kg_features: np.ndarray) -> np.ndarray:
        """
        合并轨迹特征和知识图谱特征（可选方法）

        Args:
            trajectory_features: (N, 9) 轨迹特征
            kg_features: (N, 11) KG 特征

        Returns:
            combined: (N, 20) 合并后的特征
        """
        return np.concatenate([trajectory_features, kg_features], axis=1)