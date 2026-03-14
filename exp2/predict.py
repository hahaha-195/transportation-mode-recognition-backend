"""
Exp2 预测器 (独立版)
只需要模型文件，不需要任何额外的 pkl 缓存
"""
import torch
import numpy as np
import os
from src.model import TransportationModeClassifier


class TransportationPredictorExp2:
    def __init__(self, checkpoint_path="checkpoints/exp2_model.pth"):
        """
        初始化实验二预测器
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

        # 初始化模型架构 (Exp2: 轨迹 9 维 + KG 11 维)
        self.model = TransportationModeClassifier(
            trajectory_feature_dim=config['trajectory_feature_dim'],
            kg_feature_dim=config['kg_feature_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_classes=config['num_classes'],
            dropout=config.get('dropout', 0.3)
        ).to(self.device)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        print(f"✅ Exp2 模型加载成功！")
        print(f"📊 特征配置: 轨迹={config['trajectory_feature_dim']}维, KG={config['kg_feature_dim']}维")
        print(f"🏷️  支持类别: {list(self.class_names)}")

    def predict(self, traj_features, kg_features):
        """
        输入参数:
            traj_features: (seq_len, 9) 的轨迹特征矩阵
            kg_features: (11,) 的知识图谱特征向量
        返回:
            pred_label: 预测的交通方式字符串
            confidence: 置信度分数
        """
        # 1. 轨迹特征维度处理 (batch_size, seq_len, 9)
        if traj_features.ndim == 2:
            traj_features = np.expand_dims(traj_features, axis=0)

        # 2. KG 特征维度处理 (batch_size, 1, 11)
        # 增加维度以满足模型内部 kg_out[:, -1, :] 的索引需求
        if kg_features.ndim == 1:
            kg_features = np.expand_dims(np.expand_dims(kg_features, axis=0), axis=1)
        elif kg_features.ndim == 2:
            kg_features = np.expand_dims(kg_features, axis=1)

        # 3. 转换为 Tensor 并移动到设备
        traj_tensor = torch.FloatTensor(traj_features).to(self.device)
        kg_tensor = torch.FloatTensor(kg_features).to(self.device)

        # 4. 执行推理
        with torch.no_grad():
            logits = self.model(traj_tensor, kg_tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)

        # 5. 获取结果
        pred_label = self.label_encoder.inverse_transform(pred_idx.cpu().numpy())[0]
        score = confidence.cpu().item()

        return pred_label, score


# ==========================================
# 示例用法
# ==========================================
if __name__ == "__main__":
    predictor = TransportationPredictorExp2()

    # 模拟输入：
    # 轨迹特征: 长度50, 维度9
    dummy_traj = np.random.randn(50, 9)
    # KG特征: 维度11
    dummy_kg = np.random.randn(11)

    label, score = predictor.predict(dummy_traj, dummy_kg)

    print("\n" + "=" * 40)
    print(f"【实验二预测结论】")
    print(f"识别模式: {label}")
    print(f"置信水平: {score:.4%}")
    print("=" * 40)