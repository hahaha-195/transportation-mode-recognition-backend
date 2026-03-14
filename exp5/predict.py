"""
Exp5 预测器 (独立版)
只需要模型文件，不需要任何额外的 pkl 缓存
"""
import torch
import numpy as np
import os
from src.model_weak_supervision import WeaklySupervisedContextModel


class TransportationPredictorExp5:
    def __init__(self, checkpoint_path="checkpoints/exp5_model.pth"):
        """
        初始化实验五预测器
        :param checkpoint_path: 训练好的模型权重路径（包含 label_encoder）
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 加载模型 Checkpoint（包含 label_encoder）
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"❌ 未找到模型文件: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # 从 checkpoint 中获取 label_encoder
        self.label_encoder = ckpt['label_encoder']
        self.class_names = self.label_encoder.classes_
        config = ckpt['model_config']

        # 初始化模型架构 (Exp5: 轨迹 9 维 + 增强 KG 15 维 + 天气 12 维)
        # 弱监督约束策略：KG和天气特征不直接参与分类，而是通过一致性损失约束轨迹表示
        self.model = WeaklySupervisedContextModel(
            trajectory_feature_dim=config.get('trajectory_feature_dim', 9),
            kg_feature_dim=config.get('kg_feature_dim', 15),
            weather_feature_dim=config.get('weather_feature_dim', 12),
            hidden_dim=config.get('hidden_dim', 128),
            num_layers=config.get('num_layers', 2),
            num_classes=config.get('num_classes', 7),
            dropout=config.get('dropout', 0.3),
            context_loss_type=config.get('context_loss_type', 'mse'),
            context_loss_weight=config.get('context_loss_weight', 0.05)
        ).to(self.device)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        print(f"✅ Exp5 模型加载成功！")
        print(f"📊 特征配置: 轨迹={config.get('trajectory_feature_dim', 9)}维, "
              f"KG={config.get('kg_feature_dim', 15)}维, "
              f"天气={config.get('weather_feature_dim', 12)}维")
        print(f"🔗 融合策略: 弱监督约束 (context_loss_weight={config.get('context_loss_weight', 0.05)})")
        print(f"🏷️  支持类别: {list(self.class_names)}")

    def predict(self, traj_features, kg_features, weather_features):
        """
        输入参数:
            traj_features: (seq_len, 9) 的轨迹特征矩阵
            kg_features: (15,) 的增强知识图谱特征向量
            weather_features: (12,) 的天气特征向量
        返回:
            pred_label: 预测的交通方式字符串
            confidence: 置信度分数
        """
        # 1. 轨迹特征维度处理 -> (1, seq_len, 9)
        if traj_features.ndim == 2:
            traj_features = np.expand_dims(traj_features, axis=0)

        # 2. KG 特征维度处理 -> (1, 1, 15)
        if kg_features.ndim == 1:
            kg_features = np.expand_dims(np.expand_dims(kg_features, axis=0), axis=1)
        elif kg_features.ndim == 2:
            kg_features = np.expand_dims(kg_features, axis=1)

        # 3. 天气特征维度处理 -> (1, 1, 12)
        if weather_features.ndim == 1:
            weather_features = np.expand_dims(np.expand_dims(weather_features, axis=0), axis=1)
        elif weather_features.ndim == 2:
            weather_features = np.expand_dims(weather_features, axis=1)

        # 4. 转换为 Tensor
        traj_tensor = torch.FloatTensor(traj_features).to(self.device)
        kg_tensor = torch.FloatTensor(kg_features).to(self.device)
        weather_tensor = torch.FloatTensor(weather_features).to(self.device)

        # 5. 推理
        with torch.no_grad():
            logits = self.model(traj_tensor, kg_tensor, weather_tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)

        # 6. 映射回标签
        pred_label = self.label_encoder.inverse_transform(pred_idx.cpu().numpy())[0]
        score = confidence.cpu().item()

        return pred_label, score


# ==========================================
# 示例用法
# ==========================================
if __name__ == "__main__":
    predictor = TransportationPredictorExp5()

    # 模拟输入：
    dummy_traj = np.random.randn(50, 9)    # 50个点的轨迹
    dummy_kg = np.random.randn(15)         # 15维增强KG特征
    dummy_weather = np.random.randn(12)    # 12维天气特征

    label, score = predictor.predict(dummy_traj, dummy_kg, dummy_weather)

    print("\n" + "=" * 40)
    print(f"【实验五预测结论】")
    print(f"识别模式: {label}")
    print(f"置信水平: {score:.4%}")
    print("=" * 40)
