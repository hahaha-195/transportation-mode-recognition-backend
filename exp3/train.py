"""
训练脚本 (Exp3)
基于 Exp2 成功架构，新增:
1. 增强 KG 特征 (15维)
2. 缓存版本控制
3. 跨机器训练支持
✅ 已集成快速模式支持
"""
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
import hashlib
import json
from datetime import datetime
import numpy as np

# ===== ✅ 修改 1: 文件开头添加导入 =====
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import BaseGeoLifePreprocessor, Exp3DataAdapter
# ===== 新增结束 =====

from src.data_preprocessing import GeoLifeDataLoader, OSMDataLoader, preprocess_trajectory_segments
from src.feature_extraction import FeatureExtractor
from src.model import TransportationModeClassifier
from src.knowledge_graph import EnhancedTransportationKG

# ========================== 特征维度常量 ==========================
TRAJECTORY_FEATURE_DIM = 9
KG_FEATURE_DIM = 15  # Exp3: 11 → 15
# ==================================================================

# ========================== 缓存配置 ==========================
CACHE_VERSION = "v1"
CACHE_DIR = 'cache'
KG_CACHE_PATH = os.path.join(CACHE_DIR, f'kg_data_{CACHE_VERSION}.pkl')
GRID_CACHE_PATH = os.path.join(CACHE_DIR, f'grid_cache_{CACHE_VERSION}.pkl')
PROCESSED_FEATURE_CACHE_PATH = os.path.join(CACHE_DIR, f'processed_features_{CACHE_VERSION}.pkl')
META_CACHE_PATH = os.path.join(CACHE_DIR, 'cache_meta.json')
os.makedirs(CACHE_DIR, exist_ok=True)


def compute_file_hash(filepath: str) -> str:
    """计算文件 MD5 哈希"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def save_cache_metadata(osm_path: str, geolife_root: str,
                        num_segments: int, label_encoder: LabelEncoder):
    """保存缓存元数据"""
    meta = {
        "version": CACHE_VERSION,
        "experiment": "exp3",
        "created_at": datetime.now().isoformat(),
        "osm_file": osm_path,
        "osm_file_hash": compute_file_hash(osm_path),
        "geolife_root": geolife_root,
        "kg_feature_dim": KG_FEATURE_DIM,
        "trajectory_feature_dim": TRAJECTORY_FEATURE_DIM,
        "total_feature_dim": TRAJECTORY_FEATURE_DIM + KG_FEATURE_DIM,
        "num_segments": num_segments,
        "num_classes": len(label_encoder.classes_),
        "classes": label_encoder.classes_.tolist()
    }

    with open(META_CACHE_PATH, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"✓ 缓存元数据已保存: {META_CACHE_PATH}")


def validate_cache(osm_path: str) -> bool:
    """验证缓存是否有效"""
    if not os.path.exists(META_CACHE_PATH):
        return False

    try:
        with open(META_CACHE_PATH, 'r') as f:
            meta = json.load(f)

        if meta.get('version') != CACHE_VERSION:
            print(f"⚠️  缓存版本不匹配: {meta.get('version')} != {CACHE_VERSION}")
            return False

        if meta.get('experiment') != 'exp3':
            print(f"⚠️  缓存实验类型不匹配: {meta.get('experiment')} != exp3")
            return False

        current_hash = compute_file_hash(osm_path)
        if meta.get('osm_file_hash') != current_hash:
            print(f"⚠️  OSM 文件已更改，缓存失效")
            return False

        print(f"✓ 缓存验证通过 (版本: {CACHE_VERSION})")
        return True

    except Exception as e:
        print(f"⚠️  缓存验证失败: {e}")
        return False


class TrajectoryDataset(Dataset):
    """轨迹数据集"""

    def __init__(self, all_features_and_labels):
        self.data = all_features_and_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        trajectory_features, kg_features, label_encoded = self.data[idx]

        trajectory_tensor = torch.FloatTensor(trajectory_features)
        kg_tensor = torch.FloatTensor(kg_features)
        label_tensor = torch.LongTensor([label_encoded])[0]

        return trajectory_tensor, kg_tensor, label_tensor


# ============================================================
# Data loading (✅ 修改 2: 更新 load_data 函数集成快速模式)
# ============================================================
def load_data(geolife_root: str, osm_path: str, max_users: int = None, use_base_data: bool = True, cleaning_mode: str = 'balanced'):
    """加载所有数据 (支持快速模式与三级缓存)

    Args:
        geolife_root: GeoLife数据根目录
        osm_path: OSM数据路径
        max_users: 最大用户数
        use_base_data: 是否使用预处理的基础数据
        cleaning_mode: 数据清洗模式 ('strict', 'balanced', 'gentle')
    """

    BASE_DATA_PATH = os.path.join(
        os.path.dirname(geolife_root),
        'processed/base_segments.pkl'
    )

    geolife_loader = GeoLifeDataLoader(geolife_root)
    users = geolife_loader.get_all_users()
    if max_users and max_users < len(users):
        users = users[:max_users]

    # ================= 阶段 1: 知识图谱构建 ==================
    kg = None
    if os.path.exists(KG_CACHE_PATH):
        print(f"\n========== 阶段 1: 知识图谱加载 (从缓存) ==========")
        try:
            with open(KG_CACHE_PATH, 'rb') as f:
                kg = pickle.load(f)
            print("✅ 知识图谱从缓存加载完成。")
            if os.path.exists(GRID_CACHE_PATH):
                kg.load_cache(GRID_CACHE_PATH)
        except Exception as e:
            warnings.warn(f"[WARN] KG 缓存加载失败 ({e})")
            kg = None

    if kg is None:
        print("\n========== 阶段 1: 知识图谱构建 (重建) ==========")
        osm_loader = OSMDataLoader(osm_path)
        osm_data = osm_loader.load_osm_data()
        road_network = osm_loader.extract_road_network(osm_data)
        pois = osm_loader.extract_pois(osm_data)
        transit_routes = osm_loader.extract_transit_routes(osm_data)

        kg = EnhancedTransportationKG()
        kg.build_from_osm(road_network, pois, transit_routes)

        with open(KG_CACHE_PATH, 'wb') as f:
            pickle.dump(kg, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("✅ 知识图谱缓存完成。")

    # ================= 阶段 2: 数据加载与特征提取 ==================
    all_features_and_labels = None
    label_encoder = None

    if os.path.exists(PROCESSED_FEATURE_CACHE_PATH):
        print(f"\n========== 阶段 2: 特征加载 (从最终缓存) ==========")
        try:
            with open(PROCESSED_FEATURE_CACHE_PATH, 'rb') as f:
                all_features_and_labels, label_encoder = pickle.load(f)
            print(f"✅ 预提取特征加载完成: {len(all_features_and_labels)} 条记录")
            return all_features_and_labels, kg, label_encoder, {}
        except Exception:
            pass

    processed_segments = None
    cleaning_stats = {}

    # ✅ 快速模式：使用基础数据
    if use_base_data and os.path.exists(BASE_DATA_PATH):
        print(f"\n{'='*80}")
        print(f"阶段 2: 使用预处理的基础数据（快速模式 - 清洗模式: {cleaning_mode}）")
        print(f"{'='*80}\n")

        base_segments = BaseGeoLifePreprocessor.load_from_cache(BASE_DATA_PATH)
        adapter = Exp3DataAdapter(target_length=50, enable_cleaning=True, cleaning_mode=cleaning_mode)
        processed_segments = adapter.process_segments(base_segments)
        cleaning_stats = adapter.get_cleaning_stats()

    # 传统模式：从头处理
    else:
        if use_base_data:
            print(f"\n⚠️  基础数据不存在，使用传统模式")

        print("\n========== 阶段 2: 加载轨迹数据 (传统模式) ==========")
        all_segments = []
        for user_id in tqdm(users, desc="[用户加载]"):
            labels = geolife_loader.load_labels(user_id)
            if labels.empty: continue
            trajectory_dir = os.path.join(geolife_root, f"Data/{user_id}/Trajectory")
            if not os.path.exists(trajectory_dir): continue

            for traj_file in os.listdir(trajectory_dir):
                if not traj_file.endswith('.plt'): continue
                try:
                    trajectory = geolife_loader.load_trajectory(os.path.join(trajectory_dir, traj_file))
                    all_segments.extend(geolife_loader.segment_trajectory(trajectory, labels))
                except: continue

        processed_segments = preprocess_trajectory_segments(all_segments, min_length=10)
        valid_modes = {'Walk', 'Bike', 'Bus', 'Car & taxi', 'Train', 'Subway', 'Airplane'}
        processed_segments = [(t, l) for t, l in processed_segments if l in valid_modes]

    # 特征提取阶段
    if not processed_segments:
        print("错误: 没有可用轨迹段")
        return [], kg, None

    all_labels_str = [label for _, label in processed_segments]
    label_encoder = LabelEncoder().fit(all_labels_str)

    print("\n========== 阶段 3: 特征提取 ==========")
    feature_extractor = FeatureExtractor(kg)
    all_features_and_labels = []

    for trajectory, label_str in tqdm(processed_segments, desc="[Exp3 特征提取]"):
        try:
            trajectory_features, kg_features = feature_extractor.extract_features(trajectory)
            label_encoded = label_encoder.transform([label_str])[0]
            all_features_and_labels.append((trajectory_features, kg_features, label_encoded))
        except Exception:
            continue

    # 保存各级缓存
    with open(PROCESSED_FEATURE_CACHE_PATH, 'wb') as f:
        pickle.dump((all_features_and_labels, label_encoder, cleaning_stats), f, protocol=pickle.HIGHEST_PROTOCOL)
    kg.save_cache(GRID_CACHE_PATH)
    save_cache_metadata(osm_path, geolife_root, len(all_features_and_labels), label_encoder)

    return all_features_and_labels, kg, label_encoder, cleaning_stats


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for traj_feat, kg_feat, labels in tqdm(dataloader, desc="Training Progress", leave=True):
        traj_feat, kg_feat, labels = traj_feat.to(device), kg_feat.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(traj_feat, kg_feat)
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
        for traj_feat, kg_feat, labels in tqdm(dataloader, desc="Validation Progress", leave=True):
            traj_feat, kg_feat, labels = traj_feat.to(device), kg_feat.to(device), labels.to(device)
            logits = model(traj_feat, kg_feat)
            total_loss += criterion(logits, labels).item()
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    report = classification_report(all_labels, all_preds, target_names=label_encoder.classes_, output_dict=True, zero_division=0)
    return total_loss / len(dataloader), report, all_preds, all_labels


# ============================================================
# Main (✅ 修改 3: 更新 main 函数添加命令行参数)
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='训练交通方式识别模型 (Exp3)')
    parser.add_argument('--geolife_root', type=str, default='../data/Geolife Trajectories 1.3')
    parser.add_argument('--osm_path', type=str, default='../data/exp3.geojson')

    # ===== ✅ 新增参数 =====
    parser.add_argument('--use_base_data', action='store_true', default=True, help='使用预处理的基础数据（推荐）')
    parser.add_argument('--cleaning_mode', type=str, default='balanced',
                       choices=['strict', 'balanced', 'gentle'],
                       help='数据清洗模式: strict(严格), balanced(平衡), gentle(温和)')
    # ===== 新增结束 =====

    parser.add_argument('--max_users', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    parser.add_argument('--generate_cache_only', action='store_true', help='仅生成缓存')
    parser.add_argument('--use_cached_data', action='store_true', help='直接使用缓存数据')
    parser.add_argument('--clear_cache', action='store_true', help='清空缓存')

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    if args.clear_cache:
        for f in [KG_CACHE_PATH, GRID_CACHE_PATH, PROCESSED_FEATURE_CACHE_PATH, META_CACHE_PATH]:
            if os.path.exists(f): os.remove(f)

    # ========== 步骤1: 加载数据 ==========
    if args.use_cached_data:
        if not validate_cache(args.osm_path): return
        with open(PROCESSED_FEATURE_CACHE_PATH, 'rb') as f:
            all_features_and_labels, label_encoder = pickle.load(f)
        with open(KG_CACHE_PATH, 'rb') as f:
            kg = pickle.load(f)
    else:
        # ✅ 传递新参数
        all_features_and_labels, kg, label_encoder, cleaning_stats = load_data(
            args.geolife_root, args.osm_path, args.max_users,
            use_base_data=args.use_base_data,
            cleaning_mode=args.cleaning_mode
        )

    if args.generate_cache_only or not all_features_and_labels:
        return

    # ========== 步骤2: 训练模型 ==========
    print(f"\n开始训练 (类别: {label_encoder.classes_})")
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

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
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

            torch.save({
                'model_state_dict': model.state_dict(),
                'label_encoder': label_encoder,
                'model_config': {
                    'trajectory_feature_dim': TRAJECTORY_FEATURE_DIM, 'kg_feature_dim': KG_FEATURE_DIM,
                    'hidden_dim': args.hidden_dim, 'num_layers': args.num_layers,
                    'num_classes': len(label_encoder.classes_), 'dropout': args.dropout
                }
            }, os.path.join(args.save_dir, 'exp3_model.pth'))
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
    main()