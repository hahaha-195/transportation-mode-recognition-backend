"""
深度学习模型
结合LSTM和知识图谱特征进行交通方式识别
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class TransportationModeClassifier(nn.Module):
    """交通方式分类器 - 任务定义统一 num_classes = 7"""

    def __init__(self,
                 trajectory_feature_dim: int = 9,
                 kg_feature_dim: int = 11,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 num_classes: int = 7,
                 dropout: float = 0.3):
        """
        Args:
            trajectory_feature_dim: 轨迹特征维度
            kg_feature_dim: 知识图谱特征维度
            hidden_dim: LSTM隐藏层维度
            num_layers: LSTM层数
            num_classes: 分类类别数（任务定义统一：Walk, Bike, Bus, Car&taxi, Train, Subway, Airplane）
            dropout: Dropout比率
        """
        super(TransportationModeClassifier, self).__init__()

        self.trajectory_feature_dim = trajectory_feature_dim
        self.kg_feature_dim = kg_feature_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # 轨迹特征LSTM编码器
        # Bi-LSTM 输出维度: hidden_dim * 2
        self.trajectory_lstm = nn.LSTM(
            input_size=trajectory_feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # 知识图谱特征LSTM编码器
        # Bi-LSTM 输出维度: (hidden_dim // 2) * 2 = hidden_dim
        self.kg_lstm = nn.LSTM(
            input_size=kg_feature_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # 特征融合层
        # 融合输入维度 = (trajectory_repr: hidden_dim * 2) + (kg_repr: hidden_dim)
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
        Args:
            trajectory_features: (batch_size, seq_len, trajectory_feature_dim)
            kg_features: (batch_size, seq_len, kg_feature_dim)

        Returns:
            logits: (batch_size, num_classes)
        """

        # 轨迹特征编码
        trajectory_out, (trajectory_hidden, _) = self.trajectory_lstm(trajectory_features)
        # 使用最后一个时间步的输出
        trajectory_repr = trajectory_out[:, -1, :]  # (batch_size, hidden_dim * 2)

        # 知识图谱特征编码
        kg_out, (kg_hidden, _) = self.kg_lstm(kg_features)
        # 使用最后一个时间步的输出
        kg_repr = kg_out[:, -1, :]  # (batch_size, hidden_dim)

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


class AttentionFusionModel(nn.Module):
    """带注意力机制的特征融合模型（可选） - 基于 GeoLife 7 大类修正 num_classes = 7""" # MODIFIED

    def __init__(self,
                 trajectory_feature_dim: int = 9,
                 kg_feature_dim: int = 11,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 num_classes: int = 7, # MODIFIED: 6 -> 7
                 dropout: float = 0.3):
        super(AttentionFusionModel, self).__init__()

        # 轨迹特征编码器
        self.trajectory_lstm = nn.LSTM(
            input_size=trajectory_feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # 知识图谱特征编码器
        self.kg_lstm = nn.LSTM(
            input_size=kg_feature_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        # 特征融合
        fusion_dim = hidden_dim * 2 + hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, trajectory_features: torch.Tensor,
                kg_features: torch.Tensor) -> torch.Tensor:
        # 编码轨迹特征
        trajectory_out, _ = self.trajectory_lstm(trajectory_features)

        # 编码知识图谱特征
        kg_out, _ = self.kg_lstm(kg_features)

        # 注意力融合
        # 使用轨迹特征作为query，知识图谱特征作为key和value
        # 需要对 kg_out 进行填充使其维度与 trajectory_out 匹配
        kg_out_dim = kg_out.size(-1)
        trajectory_out_dim = trajectory_out.size(-1)

        if kg_out_dim < trajectory_out_dim:
             # 填充到 hidden_dim * 2 的维度
            kg_padded = F.pad(kg_out, (0, trajectory_out_dim - kg_out_dim))
        else:
             # 确保维度一致，这里简单截断或使用线性层也可以
            kg_padded = kg_out[:, :, :trajectory_out_dim]

        attended, _ = self.attention(trajectory_out, kg_padded, kg_padded)

        # 取最后一个时间步
        trajectory_repr = trajectory_out[:, -1, :]
        attended_repr = attended[:, -1, :]

        # 融合
        combined = torch.cat([trajectory_repr, attended_repr], dim=1)
        fused = self.fusion(combined)

        # 分类
        logits = self.classifier(fused)

        return logits