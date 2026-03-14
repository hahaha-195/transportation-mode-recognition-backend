"""
Exp2 评估脚本 (标准版 - 与 Exp4 一致)
功能：
1. 自动适配轨迹 + KG特征（9+11维）
2. 支持多路径缓存加载
3. 生成详细的分类报告、混淆矩阵和预测详情
4. 输出所有文件至 evaluation_results/ 子目录
"""
import os
import json
import torch
import numpy as np
import pandas as pd
import pickle
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# 导入模型
from src.model import TransportationModeClassifier

# 设置中文字体 (防止图片乱码)
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ------------------------------------------------------------
# 1. 数据集定义
# ------------------------------------------------------------
class DualFeatureDataset(Dataset):
    def __init__(self, segments, label_encoder):
        self.segments = segments
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        # 适配训练脚本保存的格式: (traj_feat, kg_feat, label)
        traj_features, kg_features, label = self.segments[idx]

        traj_tensor = torch.FloatTensor(traj_features)
        kg_tensor = torch.FloatTensor(kg_features)

        # 标签处理：如果是字符串则转换，如果是数字则直接读取
        if isinstance(label, (int, np.integer)):
            label_tensor = torch.LongTensor([label])[0]
        else:
            label_encoded = self.label_encoder.transform([label])[0]
            label_tensor = torch.LongTensor([label_encoded])[0]

        return traj_tensor, kg_tensor, label_tensor


def main():
    # 配置参数
    MODEL_PATH = 'checkpoints/exp2_model.pth'
    CACHE_PATH = 'cache/processed_features.pkl'
    OUTPUT_DIR = 'evaluation_results'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 备用缓存路径列表（按优先级）
    ALTERNATIVE_CACHE_PATHS = [
        'cache/processed_features_v1.pkl',
        'cache/exp2_features.pkl',
    ]

    print("\n" + "=" * 60)
    print("Exp2 模型评估 (轨迹 + 基础知识图谱)")
    print("=" * 60)
    print(f"设备: {DEVICE}")

    # 1. 加载模型
    print(f"\n[1/5] 正在加载模型: {MODEL_PATH}")

    # 尝试多个模型路径
    model_paths_to_try = [
        MODEL_PATH,
        'checkpoints/exp2_model.pth',
    ]

    model_loaded = False
    for mp in model_paths_to_try:
        if os.path.exists(mp):
            MODEL_PATH = mp
            model_loaded = True
            break

    if not model_loaded:
        print(f"❌ 错误: 找不到模型文件，尝试过以下路径:")
        for mp in model_paths_to_try:
            print(f"   - {mp}")
        return

    print(f"   ✓ 找到模型: {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    label_encoder = checkpoint['label_encoder']
    config = checkpoint['model_config']
    class_names = label_encoder.classes_

    print(f"   模型配置:")
    print(f"     - 轨迹特征维度: {config['trajectory_feature_dim']}")
    print(f"     - KG特征维度: {config['kg_feature_dim']}")
    print(f"     - 隐藏层维度: {config['hidden_dim']}")
    print(f"     - 层数: {config['num_layers']}")
    print(f"     - 类别数: {config['num_classes']}")
    print(f"     - 类别: {list(class_names)}")

    # 动态初始化模型 (Exp2: 9 + 11 维)
    model = TransportationModeClassifier(
        trajectory_feature_dim=config['trajectory_feature_dim'],
        kg_feature_dim=config['kg_feature_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_classes=config['num_classes'],
        dropout=config.get('dropout', 0.3)
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("   ✓ 模型加载完成")

    # 2. 加载特征缓存
    print(f"\n[2/5] 正在加载特征缓存...")

    # 尝试多个缓存路径
    cache_paths_to_try = [CACHE_PATH] + ALTERNATIVE_CACHE_PATHS
    cache_loaded = False

    for cp in cache_paths_to_try:
        if os.path.exists(cp):
            CACHE_PATH = cp
            cache_loaded = True
            break

    if not cache_loaded:
        print(f"❌ 错误: 找不到特征缓存，尝试过以下路径:")
        for cp in cache_paths_to_try:
            print(f"   - {cp}")
        print("\n请先运行 train.py 生成特征缓存。")
        return

    print(f"   ✓ 找到缓存: {CACHE_PATH}")

    with open(CACHE_PATH, 'rb') as f:
        # 训练脚本保存的是 (all_features_and_labels, label_encoder, cleaning_stats)
        cache_data = pickle.load(f)
        if len(cache_data) == 3:
            all_features, cached_label_encoder, cleaning_stats = cache_data
        else:
            all_features, cached_label_encoder = cache_data
            cleaning_stats = {}

    print(f"   ✓ 加载完成: {len(all_features)} 个样本")

    # 显示清洗统计
    if cleaning_stats:
        print(f"\n{'=' * 60}")
        print("数据清洗统计")
        print(f"{'=' * 60}")
        before = cleaning_stats.get('before', {})
        after = cleaning_stats.get('after', {})
        cleaner = cleaning_stats.get('cleaner', {})

        if before:
            print(f"\n第一阶段（基础预处理）:")
            print(f"  输入轨迹段数: {before.get('total_segments', 0)}")
            print(f"  输入点数: {before.get('total_points', 0)}")

        if after:
            print(f"\n第二阶段（深度清洗）:")
            print(f"  有效轨迹段数: {after.get('valid_segments', 0)}")
            print(f"  第一阶段丢弃: {after.get('stage1_discarded', 0)}")
            print(f"  第二阶段丢弃: {after.get('stage2_discarded', 0)}")
            print(f"  总丢弃: {after.get('total_discarded', 0)}")
            print(f"  保留率: {after.get('retention_rate', 0):.2%}")

        if cleaner:
            print(f"\n清洗操作详情:")
            print(f"  物理异常修复: {cleaner.get('physical_anomalies_fixed', 0)}")
            print(f"  时间间隔插值: {cleaner.get('time_gaps_filled', 0)}")
            print(f"  轨迹平滑优化: {cleaner.get('trajectory_smoothed', 0)}")
            print(f"  方向异常修正: {cleaner.get('bearing_anomalies_fixed', 0)}")

        discard_reasons = cleaning_stats.get('discard_reasons', {})
        if discard_reasons:
            print(f"\n丢弃原因分布:")
            for reason, count in discard_reasons.items():
                print(f"  {reason}: {count}")

        print(f"{'=' * 60}\n")

    # 3. 准备测试数据
    print(f"\n[3/5] 正在准备测试数据...")
    dataset = DualFeatureDataset(all_features, cached_label_encoder)
    labels_for_stratify = [item[2] for item in all_features]

    _, test_indices = train_test_split(
        range(len(dataset)),
        test_size=0.2,
        random_state=42,
        stratify=labels_for_stratify
    )

    test_loader = DataLoader(
        Subset(dataset, test_indices),
        batch_size=64,
        shuffle=False,
        num_workers=0  # Windows 兼容
    )
    print(f"   ✓ 测试集大小: {len(test_indices)} 个样本")

    # 4. 执行推理
    print(f"\n[4/5] 正在进行模型推理...")
    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        for traj, kg, labels in tqdm(test_loader, desc="Evaluation Progress", leave=True):
            try:
                traj, kg, labels = traj.to(DEVICE), kg.to(DEVICE), labels.to(DEVICE)

                logits = model(traj, kg)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                y_probs.extend(probs.cpu().numpy())
            except Exception as e:
                print(f"   ⚠️ 推理异常: {e}")
                continue

    # 转换为 Numpy 数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    # 5. 生成评估报告
    print(f"\n[5/5] 正在生成评估报告...")

    # 打印分类报告
    print("\n" + "=" * 60)
    print("分类报告")
    print("=" * 60)
    report_text = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
        digits=4
    )
    print(report_text)

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 文件 1: JSON 报告
    report_dict = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    json_path = os.path.join(OUTPUT_DIR, 'evaluation_report.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report_dict, f, indent=4, ensure_ascii=False)
    print(f"   ✓ 保存: {json_path}")

    # 文件 2: CSV 预测结果
    conf_list = [float(y_probs[i, p]) for i, p in enumerate(y_pred)]
    csv_path = os.path.join(OUTPUT_DIR, 'predictions_exp2.csv')
    pd.DataFrame({
        'true_label': [class_names[i] for i in y_true],
        'pred_label': [class_names[i] for i in y_pred],
        'confidence': conf_list,
        'correct': y_true == y_pred
    }).to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"   ✓ 保存: {csv_path}")

    # 文件 3: 混淆矩阵图
    try:
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=class_names, yticklabels=class_names
        )
        plt.title('Exp2 Confusion Matrix (Trajectory + Basic KG)', fontsize=14)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=300)
        plt.close()
        print(f"   ✓ 保存: {cm_path}")
    except Exception as e:
        print(f"   ⚠️ 混淆矩阵图生成失败: {e}")

    # 文件 4: 各类别 F1-Score 图
    try:
        f1_scores = [report_dict[cls]['f1-score'] for cls in class_names]
        plt.figure(figsize=(12, 6))
        bars = sns.barplot(x=list(class_names), y=f1_scores, color='mediumslateblue')
        plt.title('Exp2 F1-Score by Transportation Mode', fontsize=14)
        plt.xlabel('Transportation Mode')
        plt.ylabel('F1-Score')
        plt.ylim(0, 1.0)
        for i, v in enumerate(f1_scores):
            plt.text(i, v + 0.02, f"{v:.3f}", ha='center', fontsize=10)
        plt.xticks(rotation=45)
        plt.tight_layout()
        f1_path = os.path.join(OUTPUT_DIR, 'per_class_f1_scores.png')
        plt.savefig(f1_path, dpi=300)
        plt.close()
        print(f"   ✓ 保存: {f1_path}")
    except Exception as e:
        print(f"   ⚠️ F1-Score 图生成失败: {e}")

    # 文件 5: 错误分析
    try:
        errors_df = pd.DataFrame({
            'true_label': [class_names[i] for i in y_true],
            'pred_label': [class_names[i] for i in y_pred],
            'confidence': conf_list
        })
        errors_df = errors_df[errors_df['true_label'] != errors_df['pred_label']]
        errors_path = os.path.join(OUTPUT_DIR, 'error_analysis.csv')
        errors_df.to_csv(errors_path, index=False, encoding='utf-8-sig')
        print(f"   ✓ 保存: {errors_path} ({len(errors_df)} 个错误样本)")
    except Exception as e:
        print(f"   ⚠️ 错误分析保存失败: {e}")

    # 汇总统计
    print("\n" + "=" * 60)
    print("评估汇总")
    print("=" * 60)
    print(f"总样本数: {len(y_true)}")
    print(f"正确预测: {(y_true == y_pred).sum()}")
    print(f"错误预测: {(y_true != y_pred).sum()}")
    print(f"准确率: {report_dict['accuracy']:.4f}")
    print(f"加权 F1: {report_dict['weighted avg']['f1-score']:.4f}")
    print(f"宏平均 F1: {report_dict['macro avg']['f1-score']:.4f}")

    print(f"\n✅ 所有评估结果已保存至: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == '__main__':
    main()