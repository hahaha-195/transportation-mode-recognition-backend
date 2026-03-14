import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tqdm import tqdm
import pickle
import warnings
import pandas as pd
import numpy as np
from typing import List, Tuple
import torch.serialization
import sys

# ===== ✅ 修改 1：支持基础数据导入 =====
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import BaseGeoLifePreprocessor, Exp2DataAdapter
# =====================================

try:
    from src.data_preprocessing import GeoLifeDataLoader, OSMDataLoader, preprocess_trajectory_segments
    from src.feature_extraction import FeatureExtractor
    from src.model import TransportationModeClassifier
    from src.knowledge_graph import TransportationKnowledgeGraph
except ImportError:
    pass

# 特征维度常量
TRAJECTORY_FEATURE_DIM = 9
KG_FEATURE_DIM = 11
FIXED_SEQUENCE_LENGTH = 50

# 缓存配置
CACHE_DIR = 'cache'
KG_CACHE_PATH = os.path.join(CACHE_DIR, 'kg_data.pkl')
GRID_CACHE_PATH = os.path.join(CACHE_DIR, 'grid_cache.pkl')
PROCESSED_SEGMENTS_CACHE_PATH = os.path.join(CACHE_DIR, 'processed_segments.pkl')
PROCESSED_FEATURE_CACHE_PATH = os.path.join(CACHE_DIR, 'processed_features.pkl')
os.makedirs(CACHE_DIR, exist_ok=True)

class TrajectoryDataset(Dataset):
    def __init__(self, all_features_and_labels: List[Tuple[np.ndarray, np.ndarray, int]]):
        self.data = all_features_and_labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        trajectory_features, kg_features, label_encoded = self.data[idx]
        return torch.FloatTensor(trajectory_features), torch.FloatTensor(kg_features), torch.LongTensor([label_encoded])[0]

# ===== ✅ 修改 2：load_data 函数完整更新 =====
def load_data(geolife_root: str, osm_path: str, max_users: int = None, use_base_data: bool = True, cleaning_mode: str = 'balanced'):
    """加载所有数据，实现快速模式与传统模式切换

    Args:
        geolife_root: GeoLife数据根目录
        osm_path: OSM数据路径
        max_users: 最大用户数
        use_base_data: 是否使用预处理的基础数据
        cleaning_mode: 数据清洗模式 ('strict', 'balanced', 'gentle')
    """

    BASE_DATA_PATH = os.path.join(os.path.dirname(geolife_root), 'processed/base_segments.pkl')

    # 1. 知识图谱构建 (保持原有逻辑)
    kg = None
    if os.path.exists(KG_CACHE_PATH):
        print(f"\n========== 阶段 1: 知识图谱加载 (从缓存) ==========")
        try:
            with open(KG_CACHE_PATH, 'rb') as f:
                with torch.serialization.safe_globals({LabelEncoder: LabelEncoder}):
                    kg = pickle.load(f)
            if os.path.exists(GRID_CACHE_PATH): kg.load_cache(GRID_CACHE_PATH)
        except Exception as e:
            kg = None

    if kg is None:
        print("\n========== 阶段 1: 知识图谱构建 (重建) ==========")
        osm_loader = OSMDataLoader(osm_path)
        osm_data = osm_loader.load_osm_data()
        kg = TransportationKnowledgeGraph()
        kg.build_from_osm(osm_loader.extract_road_network(osm_data), osm_loader.extract_pois(osm_data))
        with open(KG_CACHE_PATH, 'wb') as f:
            pickle.dump(kg, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 2. 数据准备
    # A. 最终特征缓存检查
    if os.path.exists(PROCESSED_FEATURE_CACHE_PATH):
        print(f"\n========== 阶段 2: 最终特征加载 (从缓存) ==========")
        try:
            with open(PROCESSED_FEATURE_CACHE_PATH, 'rb') as f:
                with torch.serialization.safe_globals({LabelEncoder: LabelEncoder}):
                    all_features_and_labels, label_encoder = pickle.load(f)
            return all_features_and_labels, kg, label_encoder, {}
        except Exception:
            pass

    processed_segments = None
    label_encoder = None
    cleaning_stats = {}

    # B. ✅ 快速模式逻辑
    if use_base_data and os.path.exists(BASE_DATA_PATH):
        print(f"\n========== 阶段 2: 使用预处理基础数据 (快速模式 - 清洗模式: {cleaning_mode}) ==========")
        base_segments = BaseGeoLifePreprocessor.load_from_cache(BASE_DATA_PATH)
        adapter = Exp2DataAdapter(target_length=FIXED_SEQUENCE_LENGTH, enable_cleaning=True, cleaning_mode=cleaning_mode)
        processed_segments = adapter.process_segments(base_segments)
        cleaning_stats = adapter.get_cleaning_stats()

        all_labels_str = [label for _, label in processed_segments]
        label_encoder = LabelEncoder().fit(all_labels_str)
        print(f"✅ 基础数据适配完成: {len(processed_segments)} 个段")

    # C. 传统模式逻辑
    else:
        if use_base_data: print(f"⚠️ 基础数据不存在，回退传统模式")
        if os.path.exists(PROCESSED_SEGMENTS_CACHE_PATH):
            print(f"\n========== 阶段 2.1: 轨迹段加载 (从缓存) ==========")
            with open(PROCESSED_SEGMENTS_CACHE_PATH, 'rb') as f:
                with torch.serialization.safe_globals({LabelEncoder: LabelEncoder}):
                    processed_segments, label_encoder = pickle.load(f)

        if processed_segments is None:
            print("\n========== 阶段 2.1: 轨迹段加载 (从原始文件) ==========")
            geolife_loader = GeoLifeDataLoader(geolife_root)
            users = geolife_loader.get_all_users()
            if max_users: users = users[:max_users]

            all_segments = []
            for user_id in tqdm(users, desc="[用户加载]"):
                labels = geolife_loader.load_labels(user_id)
                if labels.empty: continue
                trajectory_dir = os.path.join(geolife_root, f"Data/{user_id}/Trajectory")
                for traj_file in os.listdir(trajectory_dir):
                    if not traj_file.endswith('.plt'): continue
                    try:
                        traj = geolife_loader.load_trajectory(os.path.join(trajectory_dir, traj_file))
                        all_segments.extend(geolife_loader.segment_trajectory(traj, labels))
                    except: continue

            processed_segments = preprocess_trajectory_segments(all_segments, min_length=10)
            final_seven_modes = {'Walk', 'Bike', 'Bus', 'Car & taxi', 'Train', 'Airplane', 'Other'}
            processed_segments = [(t, l) for t, l in processed_segments if l in final_seven_modes]

            label_encoder = LabelEncoder().fit([l for _, l in processed_segments])
            with open(PROCESSED_SEGMENTS_CACHE_PATH, 'wb') as f:
                pickle.dump((processed_segments, label_encoder), f, protocol=pickle.HIGHEST_PROTOCOL)

    # 3. 特征提取 (阶段 D)
    print("\n========== 2.2: 特征提取 ==========")
    feature_extractor = FeatureExtractor(kg)
    all_features_and_labels = []
    for trajectory, label_str in tqdm(processed_segments, desc="[特征提取]"):
        try:
            trajectory_features, kg_features = feature_extractor.extract_features(trajectory)
            label_encoded = label_encoder.transform([label_str])[0]
            all_features_and_labels.append((trajectory_features, kg_features, label_encoded))
        except: continue

    with open(PROCESSED_FEATURE_CACHE_PATH, 'wb') as f:
        pickle.dump((all_features_and_labels, label_encoder, cleaning_stats), f, protocol=pickle.HIGHEST_PROTOCOL)
    kg.save_cache(GRID_CACHE_PATH)
    return all_features_and_labels, kg, label_encoder, cleaning_stats

# ... (train_epoch 和 evaluate 函数保持不变) ...

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for traj_f, kg_f, labels in tqdm(dataloader, desc="Training Progress", leave=True):
        traj_f, kg_f, labels = traj_f.to(device), kg_f.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(traj_f, kg_f)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (torch.argmax(logits, dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(dataloader), correct / total

def evaluate(model, dataloader, criterion, device, label_encoder):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    with torch.no_grad():
        for traj_f, kg_f, labels in tqdm(dataloader, desc="Validation Progress", leave=True):
            traj_f, kg_f, labels = traj_f.to(device), kg_f.to(device), labels.to(device)
            logits = model(traj_f, kg_f)
            total_loss += criterion(logits, labels).item()
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    report = classification_report(all_labels, all_preds, target_names=label_encoder.classes_, output_dict=True, zero_division=0)
    return total_loss / len(dataloader), report, all_preds, all_labels

# ===== ✅ 修改 3：main 函数参数接入 =====
def main():
    parser = argparse.ArgumentParser(description='训练交通方式识别模型 (Exp2 优化版)')
    parser.add_argument('--geolife_root', type=str, default='../data/Geolife Trajectories 1.3')
    parser.add_argument('--osm_path', type=str, default='../data/exp2.geojson')

    # 新增参数
    parser.add_argument('--use_base_data', action='store_true', default=True, help='使用预处理的基础数据')
    parser.add_argument('--cleaning_mode', type=str, default='balanced',
                       choices=['strict', 'balanced', 'gentle'],
                       help='数据清洗模式: strict(严格), balanced(平衡), gentle(温和)')

    parser.add_argument('--max_users', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--clear_cache', action='store_true')

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    if args.clear_cache:
        for f in [KG_CACHE_PATH, GRID_CACHE_PATH, PROCESSED_SEGMENTS_CACHE_PATH, PROCESSED_FEATURE_CACHE_PATH]:
            if os.path.exists(f): os.remove(f)

    # 传递 use_base_data 参数
    all_features_and_labels, kg, label_encoder, cleaning_stats = load_data(
        args.geolife_root, args.osm_path, args.max_users,
        use_base_data=args.use_base_data,
        cleaning_mode=args.cleaning_mode
    )

    if not all_features_and_labels: return

    # ========================================================
    # ✅ 数据划分：一次性划分 70% 训练 / 10% 验证 / 20% 测试
    # ========================================================
    dataset = TrajectoryDataset(all_features_and_labels)

    print(f"\n✅ 数据集大小:")
    print(f"  总样本数: {len(dataset)}")
    print(f"  特征样本数: {len(all_features_and_labels)}")

    # ========================================================
    # ✅ 数据划分：一次性划分 70% 训练 / 10% 验证 / 20% 测试
    # ========================================================
    all_indices = np.arange(len(dataset))
    labels_stratify = [label_encoder.inverse_transform([label_encoded])[0] for _, _, label_encoded in all_features_and_labels]

    # 第一次划分：70% 训练 / 30% 临时（验证+测试）
    train_indices, temp_indices = train_test_split(
        all_indices,
        test_size=0.3,
        random_state=42,
        stratify=labels_stratify
    )

    # 第二次划分：从30%临时中划分出 10% 验证 / 20% 测试
    temp_labels = [labels_stratify[i] for i in temp_indices]
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=0.6667,
        random_state=42,
        stratify=temp_labels
    )

    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    test_loader = DataLoader(
        torch.utils.data.Subset(dataset, test_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    print(f"\n✅ 数据加载完成:")
    print(f"  Train: {len(train_indices)} 样本")
    print(f"  Val:   {len(val_indices)} 样本")
    print(f"  Test:  {len(test_indices)} 样本")
    print(f"  训练批次总数: {len(train_loader)}")
    print(f"  验证批次总数: {len(val_loader)}")

    print(f"\n类别分布:")
    for cls in label_encoder.classes_:
        train_count = sum(1 for i in train_indices if labels_stratify[i] == cls)
        val_count = sum(1 for i in val_indices if labels_stratify[i] == cls)
        test_count = sum(1 for i in test_indices if labels_stratify[i] == cls)
        print(f"  {cls:15s}: Train={train_count}, Val={val_count}, Test={test_count}")

    model = TransportationModeClassifier(
        TRAJECTORY_FEATURE_DIM, KG_FEATURE_DIM, args.hidden_dim, args.num_layers, len(label_encoder.classes_), args.dropout
    ).to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss()

    # ========================================================
    # ✅ Early Stopping 配置
    # ========================================================
    patience = 10
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, args.device)

        val_loss, report, _, _ = evaluate(model, val_loader, criterion, args.device, label_encoder)
        val_acc = report['accuracy']

        # 在训练循环结束后再打印上一轮指标汇总
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0

            torch.save({'model_state_dict': model.state_dict(), 'label_encoder': label_encoder, 'model_config': {
                'trajectory_feature_dim': TRAJECTORY_FEATURE_DIM, 'kg_feature_dim': KG_FEATURE_DIM,
                'hidden_dim': args.hidden_dim, 'num_layers': args.num_layers, 'num_classes': len(label_encoder.classes_), 'dropout': args.dropout
            }}, os.path.join(args.save_dir, 'exp2_model.pth'))
            print("✓ 保存最佳模型（基于验证集）")
        else:
            epochs_no_improve += 1
            print(f"⏳ 验证损失未改善: {epochs_no_improve}/{patience}")

            if epochs_no_improve >= patience:
                print(f"\n🛑 Early stopping 触发（patience={patience}）")
                break

    # ========================================================
    # ✅ 在测试集上进行最终评估
    # ========================================================
    print("\n" + "=" * 80)
    print("最终测试集评估")
    print("=" * 80)

    test_loss, test_report, _, _ = evaluate(model, test_loader, criterion, args.device, label_encoder)
    test_acc = test_report['accuracy']

    print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    print("\n各类别详细指标:")
    for cls in label_encoder.classes_:
        if cls in test_report:
            metrics = test_report[cls]
            print(f"  {cls:15s}: P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}")

if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)
    try:
        with torch.serialization.safe_globals({LabelEncoder: LabelEncoder}):
            main()
    except Exception as e:
        print(f"Error: {e}")