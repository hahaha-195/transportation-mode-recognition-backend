"""
深度学习模型 (Exp5 - 弱监督上下文表示增强)
核心思想：GTA-Seg - 上下文特征仅用于改善轨迹编码器表示，不参与分类决策

与Exp4的关键区别：
- Exp4: trajectory + KG + weather 硬拼接 → classifier
- Exp5: trajectory → classifier，KG+weather 仅作为 context encoder 约束
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class WeaklySupervisedContextModel(nn.Module):
    """弱监督上下文表示增强模型 (Exp5)

    核心设计：
    1. 轨迹编码器：独立编码轨迹特征，输出轨迹表示
    2. 上下文编码器：编码KG+天气特征，输出上下文表示
    3. 分类器：仅接收轨迹表示，不受上下文直接影响
    4. 一致性约束：embedding-level损失约束轨迹表示与上下文表示的一致性
    """

    def __init__(self,
                 trajectory_feature_dim: int = 9,
                 kg_feature_dim: int = 15,
                 weather_feature_dim: int = 12,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 num_classes: int = 7,
                 dropout: float = 0.3,
                 context_loss_type: str = 'mse',
                 context_loss_weight: float = 0.1):
        """
        Args:
            trajectory_feature_dim: 轨迹特征维度
            kg_feature_dim: 知识图谱特征维度
            weather_feature_dim: 天气特征维度
            hidden_dim: LSTM隐藏层维度
            num_layers: LSTM层数
            num_classes: 分类类别数
            dropout: Dropout比率
            context_loss_type: 上下文损失类型 ('mse' | 'cosine' | 'combined')
            context_loss_weight: 上下文损失权重 λ
        """
        super(WeaklySupervisedContextModel, self).__init__()

        self.trajectory_feature_dim = trajectory_feature_dim
        self.kg_feature_dim = kg_feature_dim
        self.weather_feature_dim = weather_feature_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.context_loss_type = context_loss_type
        self.context_loss_weight = context_loss_weight

        # ========== 轨迹编码器 ==========
        self.trajectory_lstm = nn.LSTM(
            input_size=trajectory_feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # 轨迹表示投影层（用于与上下文表示对齐）
        self.trajectory_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ========== 上下文编码器 ==========
        # KG特征编码器
        self.kg_lstm = nn.LSTM(
            input_size=kg_feature_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # 天气特征编码器
        self.weather_lstm = nn.LSTM(
            input_size=weather_feature_dim,
            hidden_size=hidden_dim // 4,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # 上下文融合层（KG + weather）
        context_fusion_dim = hidden_dim + hidden_dim // 2
        self.context_fusion = nn.Sequential(
            nn.Linear(context_fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ========== 分类器（仅接收轨迹表示） ==========
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, trajectory_features: torch.Tensor,
                kg_features: torch.Tensor,
                weather_features: torch.Tensor,
                return_context: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            trajectory_features: (batch_size, seq_len, 9)
            kg_features: (batch_size, seq_len, 15)
            weather_features: (batch_size, seq_len, 12)
            return_context: 是否返回上下文表示（用于计算一致性损失）

        Returns:
            logits: (batch_size, num_classes) - 分类logits
            trajectory_repr: (batch_size, hidden_dim) - 轨迹表示
            context_repr: (batch_size, hidden_dim) - 上下文表示（如果return_context=True）
        """
        # ========== 轨迹编码 ==========
        trajectory_out, _ = self.trajectory_lstm(trajectory_features)
        trajectory_raw = trajectory_out[:, -1, :]  # (batch, hidden_dim * 2)
        trajectory_repr = self.trajectory_proj(trajectory_raw)  # (batch, hidden_dim)

        # ========== 上下文编码 ==========
        # KG特征编码
        kg_out, _ = self.kg_lstm(kg_features)
        kg_repr = kg_out[:, -1, :]  # (batch, hidden_dim)

        # 天气特征编码
        weather_out, _ = self.weather_lstm(weather_features)
        weather_repr = weather_out[:, -1, :]  # (batch, hidden_dim // 2)

        # 上下文融合
        context_combined = torch.cat([kg_repr, weather_repr], dim=1)
        context_repr = self.context_fusion(context_combined)  # (batch, hidden_dim)

        # ========== 分类（仅使用轨迹表示） ==========
        logits = self.classifier(trajectory_repr)

        if return_context:
            return logits, trajectory_repr, context_repr
        else:
            return logits

    def compute_context_loss(self, trajectory_repr: torch.Tensor,
                          context_repr: torch.Tensor) -> torch.Tensor:
        """
        计算embedding-level一致性损失

        Args:
            trajectory_repr: (batch_size, hidden_dim) - 轨迹表示
            context_repr: (batch_size, hidden_dim) - 上下文表示

        Returns:
            context_loss: 标量损失值
        """
        if self.context_loss_type == 'mse':
            # MSE损失：最小化轨迹表示与上下文表示的欧氏距离
            context_loss = F.mse_loss(trajectory_repr, context_repr)

        elif self.context_loss_type == 'cosine':
            # Cosine损失：最大化轨迹表示与上下文表示的余弦相似度
            cosine_sim = F.cosine_similarity(trajectory_repr, context_repr, dim=1)
            context_loss = 1 - cosine_sim.mean()

        elif self.context_loss_type == 'combined':
            # 组合损失：MSE + Cosine
            mse_loss = F.mse_loss(trajectory_repr, context_repr)
            cosine_sim = F.cosine_similarity(trajectory_repr, context_repr, dim=1)
            cosine_loss = 1 - cosine_sim.mean()
            context_loss = mse_loss + cosine_loss

        else:
            raise ValueError(f"Unknown context_loss_type: {self.context_loss_type}")

        return context_loss

    def predict(self, trajectory_features: torch.Tensor,
                kg_features: torch.Tensor,
                weather_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """预测类别和概率"""
        self.eval()
        with torch.no_grad():
            logits, _, _ = self.forward(trajectory_features, kg_features, weather_features)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        return preds, probs


class TransportationModeClassifierExp5(nn.Module):
    """Exp5兼容接口（保持与Exp4相同的接口）"""
    def __init__(self, *args, **kwargs):
        super(TransportationModeClassifierExp5, self).__init__()
        self.model = WeaklySupervisedContextModel(*args, **kwargs)

    def forward(self, trajectory_features: torch.Tensor,
                kg_features: torch.Tensor,
                weather_features: torch.Tensor) -> torch.Tensor:
        logits, _, _ = self.model.forward(trajectory_features, kg_features, weather_features)
        return logits

    def predict(self, trajectory_features: torch.Tensor,
                kg_features: torch.Tensor,
                weather_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model.predict(trajectory_features, kg_features, weather_features)