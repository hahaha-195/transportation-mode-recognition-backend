# ============================================================
# exp3/src/feature_extraction.py
# ============================================================
"""
特征提取模块 (Exp3)
维度更新: KG特征 11维 → 15维
"""
import numpy as np
from typing import Tuple
from .knowledge_graph import EnhancedTransportationKG


class FeatureExtractor:
    """特征提取器 (Exp3)"""

    def __init__(self, kg: EnhancedTransportationKG):
        self.kg = kg

    def extract_features(self, trajectory: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        提取特征

        Args:
            trajectory: (N, 9) 轨迹数组

        Returns:
            trajectory_features: (N, 9) 归一化轨迹特征
            kg_features: (N, 15) 增强KG特征
        """
        # 1. 提取和归一化轨迹特征
        trajectory_features = self._extract_trajectory_features(trajectory)

        # 2. 提取增强KG特征
        try:
            kg_features = self.kg.extract_kg_features(trajectory)
        except Exception as e:
            print(f"警告: KG 特征提取失败 ({e}). 使用零填充代替。")
            kg_features = np.zeros((trajectory.shape[0], 15), dtype=np.float32)

        # 3. 严格验证维度 (N, 15)
        # trajectory 形状为 (50, 9)，kg_features 必须为 (50, 15)
        if kg_features.ndim != 2 or kg_features.shape[1] != 15:
            # 如果被压平了，重新 reshape
            if kg_features.size == trajectory.shape[0] * 15:
                kg_features = kg_features.reshape(trajectory.shape[0], 15)
            else:
                raise ValueError(f"KG 特征维度错误：预期末尾维度 15，实际 shape 为 {kg_features.shape}")

        return trajectory_features, kg_features

    def _extract_trajectory_features(self, trajectory: np.ndarray) -> np.ndarray:
        """提取并归一化轨迹特征"""
        features = trajectory.copy()
        features = self._normalize_features(features)
        return features

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Z-score 归一化"""
        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True) + 1e-8
        normalized = (features - mean) / std
        normalized = np.clip(normalized, -5, 5)
        return normalized