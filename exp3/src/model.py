# ============================================================
# exp3/src/model.py
# ============================================================
"""
深度学习模型 (Exp3)
维度更新: kg_feature_dim = 15
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class TransportationModeClassifier(nn.Module):
    """交通方式分类器 (Exp3) - 任务定义统一 num_classes = 7"""

    def __init__(self,
                 trajectory_feature_dim: int = 9,
                 kg_feature_dim: int = 15,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 num_classes: int = 7,
                 dropout: float = 0.3):
        super(TransportationModeClassifier, self).__init__()

        self.trajectory_feature_dim = trajectory_feature_dim
        self.kg_feature_dim = kg_feature_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # 轨迹特征LSTM编码器
        self.trajectory_lstm = nn.LSTM(
            input_size=trajectory_feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # 知识图谱特征LSTM编码器
        self.kg_lstm = nn.LSTM(
            input_size=kg_feature_dim,  # 15维
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # 特征融合层
        fusion_input_dim = hidden_dim * 2 + hidden_dim
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 分类层
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, trajectory_features: torch.Tensor,
                kg_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            trajectory_features: (batch_size, seq_len, 9)
            kg_features: (batch_size, seq_len, 15)

        Returns:
            logits: (batch_size, num_classes)
        """
        # 轨迹特征编码
        trajectory_out, _ = self.trajectory_lstm(trajectory_features)
        trajectory_repr = trajectory_out[:, -1, :]

        # KG特征编码
        kg_out, _ = self.kg_lstm(kg_features)
        kg_repr = kg_out[:, -1, :]

        # 特征融合
        combined = torch.cat([trajectory_repr, kg_repr], dim=1)
        fused = self.fusion_layer(combined)

        # 分类
        logits = self.classifier(fused)

        return logits

    def predict(self, trajectory_features: torch.Tensor,
                kg_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """预测类别和概率"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(trajectory_features, kg_features)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        return preds, probs