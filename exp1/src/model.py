"""
深度学习模型
仅使用GPS轨迹特征进行交通方式识别
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class TransportationModeClassifier(nn.Module):
    """交通方式分类器（仅轨迹特征）"""
    
    def __init__(self, 
                 input_dim: int = 9,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 num_classes: int = 6,
                 dropout: float = 0.3,
                 bidirectional: bool = True):
        """
        Args:
            input_dim: 输入特征维度（轨迹特征维度）
            hidden_dim: LSTM隐藏层维度
            num_layers: LSTM层数
            num_classes: 分类类别数（walk, bike, car, bus, train, taxi）
            dropout: Dropout比率
            bidirectional: 是否使用双向LSTM
        """
        super(TransportationModeClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        
        # LSTM编码器
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # 计算LSTM输出维度
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # 特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 分类层
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_dim) 轨迹特征序列
        
        Returns:
            logits: (batch_size, num_classes) 类别logits
        """
        # LSTM编码
        lstm_out, (hidden, _) = self.lstm(x)
        
        # 使用最后一个时间步的输出
        last_output = lstm_out[:, -1, :]  # (batch_size, lstm_output_dim)
        
        # 特征提取
        features = self.feature_extractor(last_output)
        
        # 分类
        logits = self.classifier(features)
        
        return logits
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """预测类别和概率"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        return preds, probs


class CNNLSTMModel(nn.Module):
    """CNN+LSTM混合模型（可选）"""
    
    def __init__(self, 
                 input_dim: int = 9,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 num_classes: int = 6,
                 dropout: float = 0.3):
        super(CNNLSTMModel, self).__init__()
        
        # 1D CNN用于提取局部特征
        self.conv1d = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM编码器
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, input_dim)
        # 转换为 (batch_size, input_dim, seq_len) 用于CNN
        x = x.transpose(1, 2)
        
        # CNN特征提取
        x = self.conv1d(x)  # (batch_size, 32, seq_len)
        
        # 转换回 (batch_size, seq_len, 32) 用于LSTM
        x = x.transpose(1, 2)
        
        # LSTM编码
        lstm_out, _ = self.lstm(x)
        
        # 使用最后一个时间步
        last_output = lstm_out[:, -1, :]
        
        # 分类
        logits = self.classifier(last_output)
        
        return logits

