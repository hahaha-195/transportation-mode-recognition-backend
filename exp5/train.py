"""
训练脚本 (Exp5 - 弱监督上下文表示增强)
核心改进：
1. 支持三种清洗模式 (strict/balanced/gentle)
2. 清洗模式独立缓存管理
3. 完整的质量评估和统计报告
4. 弱监督上下文表示增强（GTA-Seg思想）
   - 上下文特征仅用于改善轨迹编码器表示
   - 不参与分类决策
   - embedding-level一致性损失约束
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
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
import pandas as pd

# ========================== 路径设置 (PyCharm 兼容) ==========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)
os.chdir(SCRIPT_DIR)
# ==============================================================================

# 尝试导入 common 模块（可选）
try:
    from common import BaseGeoLifePreprocessor
    HAS_COMMON = True
except ImportError:
    HAS_COMMON = False
    print("⚠️ common 模块未找到，将使用传统数据加载模式")

# 尝试导入新版适配器
from common.exp5_adapter import Exp5DataAdapter

# 导入 Exp5 的弱监督模型
from exp5.src.model_weak_supervision import WeaklySupervisedContextModel

# 导入 Exp4 的数据处理模块（复用）
from exp4.src.data_preprocessing import GeoLifeDataLoader, OSMDataLoader
from exp4.src.knowledge_graph import EnhancedTransportationKG
from exp4.src.weather_preprocessing import WeatherDataProcessor
from exp4.src.feature_extraction_weather import FeatureExtractorWithWeather

# ========================== 特征维度常量 ==========================
TRAJECTORY_FEATURE_DIM = 9
KG_FEATURE_DIM = 15
WEATHER_FEATURE_DIM = 12
TOTAL_FEATURE_DIM = TRAJECTORY_FEATURE_DIM + KG_FEATURE_DIM + WEATHER_FEATURE_DIM
FIXED_SEQUENCE_LENGTH = 50
# ==================================================================


def get_cache_paths(cleaning_mode: str = 'balanced'):
    """根据清洗模式获取缓存路径"""
    cache_version = f"v5_{cleaning_mode}"
    cache_dir = 'cache'

    return {
        'version': cache_version,
        'dir': cache_dir,
        'kg': os.path.join(cache_dir, f'kg_data_{cache_version}.pkl'),
        'grid': os.path.join(cache_dir, f'grid_cache_{cache_version}.pkl'),
        'weather': os.path.join(cache_dir, f'weather_data_{cache_version}.pkl'),
        'features': os.path.join(cache_dir, f'processed_features_weather_{cache_version}.pkl'),
        'meta': os.path.join(cache_dir, f'cache_meta_{cleaning_mode}.json')
    }


def compute_file_hash(filepath: str) -> str:
    """计算文件 MD5 哈希"""
    try:
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception:
        return "unknown"


def save_cache_metadata(cache_paths: dict, osm_path: str, weather_path: str,
                       geolife_root: str, num_segments: int,
                       label_encoder: LabelEncoder, cleaning_stats: dict,
                       cleaning_mode: str):
    """保存缓存元数据（包含清洗统计）"""
    meta = {
        "version": cache_paths['version'],
        "experiment": "exp5_cleaning",
        "cleaning_mode": cleaning_mode,
        "created_at": datetime.now().isoformat(),
        "osm_file": osm_path,
        "osm_file_hash": compute_file_hash(osm_path),
        "weather_file": weather_path,
        "weather_file_hash": compute_file_hash(weather_path),
        "geolife_root": geolife_root,
        "kg_feature_dim": KG_FEATURE_DIM,
        "trajectory_feature_dim": TRAJECTORY_FEATURE_DIM,
        "weather_feature_dim": WEATHER_FEATURE_DIM,
        "total_feature_dim": TOTAL_FEATURE_DIM,
        "num_segments": num_segments,
        "num_classes": len(label_encoder.classes_),
        "classes": label_encoder.classes_.tolist(),
        "cleaning_stats": cleaning_stats
    }

    try:
        os.makedirs(os.path.dirname(cache_paths['meta']), exist_ok=True)
        with open(cache_paths['meta'], 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"✓ 缓存元数据已保存: {cache_paths['meta']}")
    except Exception as e:
        print(f"⚠️ 元数据保存失败: {e}")


def validate_cache(cache_paths: dict, osm_path: str, weather_path: str,
                  cleaning_mode: str) -> bool:
    """验证缓存是否有效"""
    if not os.path.exists(cache_paths['meta']):
        return False

    try:
        with open(cache_paths['meta'], 'r') as f:
            meta = json.load(f)

        if meta.get('version') != cache_paths['version']:
            print(f"⚠️ 缓存版本不匹配")
            return False

        if meta.get('cleaning_mode') != cleaning_mode:
            print(f"⚠️ 清洗模式不匹配 (缓存: {meta.get('cleaning_mode')}, 当前: {cleaning_mode})")
            return False

        current_osm_hash = compute_file_hash(osm_path)
        if meta.get('osm_file_hash') != current_osm_hash:
            print(f"⚠️ OSM 文件已更改")
            return False

        current_weather_hash = compute_file_hash(weather_path)
        if meta.get('weather_file_hash') != current_weather_hash:
            print(f"⚠️ 天气文件已更改")
            return False

        print(f"✓ 缓存验证通过 (版本: {cache_paths['version']}, 模式: {cleaning_mode})")
        return True

    except Exception as e:
        print(f"⚠️ 缓存验证失败: {e}")
        return False


class TrajectoryDatasetWithWeather(Dataset):
    """轨迹数据集（含天气）- Exp5版本"""

    def __init__(self, all_features_and_labels):
        self.data = all_features_and_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        trajectory_features, kg_features, weather_features, label_encoded = self.data[idx]

        trajectory_tensor = torch.FloatTensor(
            np.nan_to_num(trajectory_features, nan=0.0, posinf=0.0, neginf=0.0)
        )
        kg_tensor = torch.FloatTensor(
            np.nan_to_num(kg_features, nan=0.0, posinf=0.0, neginf=0.0)
        )
        weather_tensor = torch.FloatTensor(
            np.nan_to_num(weather_features, nan=0.0, posinf=0.0, neginf=0.0)
        )
        label_tensor = torch.LongTensor([label_encoded])[0]

        return trajectory_tensor, kg_tensor, weather_tensor, label_tensor


def normalize_features_safe(features: np.ndarray) -> np.ndarray:
    """安全的特征归一化"""
    features = features.astype(np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    mean = np.mean(features, axis=0, keepdims=True)
    std = np.std(features, axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)

    normalized = (features - mean) / std
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    normalized = np.clip(normalized, -5, 5)

    return normalized.astype(np.float32)


def load_data(geolife_root: str, osm_path: str, weather_path: str,
              max_users: int = None, use_base_data: bool = True,
              cleaning_mode: str = 'balanced'):
    """
    加载所有数据 - Exp5版本（含第二阶段清洗）

    Args:
        cleaning_mode: 清洗模式 ('strict', 'balanced', 'gentle')
    """
    # 获取缓存路径
    cache_paths = get_cache_paths(cleaning_mode)
    os.makedirs(cache_paths['dir'], exist_ok=True)

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
    if os.path.exists(cache_paths['kg']):
        print(f"\n========== 阶段 1: 知识图谱加载 (从缓存) ==========")
        try:
            with open(cache_paths['kg'], 'rb') as f:
                kg = pickle.load(f)
            print("✅ 知识图谱从缓存加载完成")
            if os.path.exists(cache_paths['grid']):
                kg.load_cache(cache_paths['grid'])
        except Exception as e:
            warnings.warn(f"[WARN] KG 缓存加载失败 ({e})，将重新构建")
            kg = None

    if kg is None:
        print("\n========== 阶段 1: 知识图谱构建 (重建) ==========")
        try:
            osm_loader = OSMDataLoader(osm_path)
            osm_data = osm_loader.load_osm_data()
            road_network = osm_loader.extract_road_network(osm_data)
            pois = osm_loader.extract_pois(osm_data)
            transit_routes = osm_loader.extract_transit_routes(osm_data)
            kg = EnhancedTransportationKG()
            kg.build_from_osm(road_network, pois, transit_routes)
            with open(cache_paths['kg'], 'wb') as f:
                pickle.dump(kg, f, protocol=pickle.HIGHEST_PROTOCOL)
            print("✅ 知识图谱缓存完成")
        except Exception as e:
            print(f"⚠️ KG 构建失败: {e}")
            kg = EnhancedTransportationKG()

    # ================= 阶段 2: 天气数据加载 ==================
    weather_processor = None
    if os.path.exists(cache_paths['weather']):
        print(f"\n========== 阶段 2: 天气数据加载 (从缓存) ==========")
        try:
            with open(cache_paths['weather'], 'rb') as f:
                weather_processor = pickle.load(f)
            print("✅ 天气数据从缓存加载完成")
        except Exception as e:
            warnings.warn(f"[WARN] 天气缓存加载失败 ({e})")
            weather_processor = None

    if weather_processor is None:
        print(f"\n========== 阶段 2: 天气数据处理 (重建) ==========")
        try:
            weather_processor = WeatherDataProcessor(weather_path)
            weather_processor.load_and_process()
            with open(cache_paths['weather'], 'wb') as f:
                pickle.dump(weather_processor, f, protocol=pickle.HIGHEST_PROTOCOL)
            print("✅ 天气数据缓存完成")
        except Exception as e:
            print(f"⚠️ 天气数据处理失败: {e}")
            weather_processor = WeatherDataProcessor(weather_path)

    # ================= 阶段 3: 轨迹数据加载与特征提取 ==================
    all_features_and_labels = None
    label_encoder = None
    cleaning_stats = {}

    # 检查缓存是否有效
    cache_valid = validate_cache(cache_paths, osm_path, weather_path, cleaning_mode)

    if cache_valid and os.path.exists(cache_paths['features']):
        print(f"\n========== 阶段 3: 特征加载 (从缓存) ==========")
        try:
            with open(cache_paths['features'], 'rb') as f:
                all_features_and_labels, label_encoder, cleaning_stats = pickle.load(f)
            print(f"✅ 特征从缓存加载完成: {len(all_features_and_labels)} 条")
            print(f"✅ 清洗统计已加载 (模式: {cleaning_mode})")
            return all_features_and_labels, kg, weather_processor, label_encoder, cleaning_stats
        except Exception as e:
            print(f"⚠️ 特征缓存加载失败: {e}")

    processed_segments_with_time = None
    adapter = None

    # 快速模式：使用基础数据 + Exp5适配器（含第二阶段清洗）
    if use_base_data and HAS_COMMON and os.path.exists(BASE_DATA_PATH):
        print(f"\n{'='*80}")
        print(f"阶段 3: 使用预处理的基础数据（快速模式 - 清洗模式: {cleaning_mode}）")
        print(f"{'='*80}\n")

        try:
            base_segments = BaseGeoLifePreprocessor.load_from_cache(BASE_DATA_PATH)

            adapter = Exp5DataAdapter(
                target_length=FIXED_SEQUENCE_LENGTH,
                enable_cleaning=True,
                cleaning_mode=cleaning_mode
            )

            processed_segments_with_time = adapter.process_segments(base_segments)

            # 打印清洗统计
            if hasattr(adapter, 'print_cleaning_summary'):
                adapter.print_cleaning_summary()
            cleaning_stats = adapter.get_cleaning_stats()

        except Exception as e:
            print(f"⚠️ 快速模式失败: {e}，切换到传统模式")
            import traceback
            traceback.print_exc()
            processed_segments_with_time = None

    # 传统模式：从头处理
    if processed_segments_with_time is None:
        if use_base_data:
            print(f"\n⚠️ 基础数据不可用，使用传统模式")

        print("\n========== 阶段 3: 加载轨迹数据 (传统模式) ==========")
        all_segments_with_dates = []

        for user_id in tqdm(users, desc="[用户加载]"):
            try:
                labels = geolife_loader.load_labels(user_id)
                if labels.empty:
                    continue
                trajectory_dir = os.path.join(geolife_root, f"Data/{user_id}/Trajectory")
                if not os.path.exists(trajectory_dir):
                    continue

                for traj_file in os.listdir(trajectory_dir):
                    if not traj_file.endswith('.plt'):
                        continue
                    try:
                        trajectory = geolife_loader.load_trajectory(
                            os.path.join(trajectory_dir, traj_file)
                        )
                        if trajectory.empty:
                            continue
                        segments = geolife_loader.segment_trajectory(trajectory, labels)
                        for seg, label in segments:
                            if 'datetime' in seg.columns and len(seg) >= 10:
                                all_segments_with_dates.append((seg, label, seg['datetime']))
                    except Exception:
                        continue
            except Exception:
                continue

        # 传统模式下的规范化处理
        processed_segments_with_time = []
        feature_cols = [
            'latitude', 'longitude', 'speed', 'acceleration',
            'bearing_change', 'distance', 'time_diff',
            'total_distance', 'total_time'
        ]

        for segment, label, dates in tqdm(all_segments_with_dates, desc="[预处理]"):
            try:
                for col in feature_cols:
                    if col not in segment.columns:
                        segment[col] = 0.0

                features = segment[feature_cols].values.astype(np.float32)
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

                current_length = len(features)

                if current_length > FIXED_SEQUENCE_LENGTH:
                    indices = np.linspace(0, current_length - 1, FIXED_SEQUENCE_LENGTH, dtype=int)
                    features = features[indices]
                    dates_resampled = dates.iloc[indices].reset_index(drop=True)
                elif current_length < FIXED_SEQUENCE_LENGTH:
                    padding = np.zeros((FIXED_SEQUENCE_LENGTH - current_length, features.shape[1]),
                                      dtype=np.float32)
                    features = np.vstack([features, padding])
                    last_date = dates.iloc[-1] if len(dates) > 0 else pd.Timestamp.now()
                    padding_dates = pd.Series([last_date] * (FIXED_SEQUENCE_LENGTH - len(dates)))
                    dates_resampled = pd.concat([dates.reset_index(drop=True), padding_dates],
                                               ignore_index=True)
                else:
                    dates_resampled = dates.reset_index(drop=True)

                processed_segments_with_time.append((features, dates_resampled, label))
            except Exception:
                continue

    # 过滤有效类别
    valid_modes = {'Walk', 'Bike', 'Bus', 'Car & taxi', 'Train', 'Subway', 'Airplane'}
    processed_segments_with_time = [
        s for s in processed_segments_with_time if s[2] in valid_modes
    ]

    if len(processed_segments_with_time) == 0:
        raise ValueError("没有有效的轨迹数据！请检查数据路径和格式。")

    # 标签编码
    all_labels = [s[2] for s in processed_segments_with_time]
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)

    print(f"\n类别分布:")
    for cls in label_encoder.classes_:
        count = sum(1 for l in all_labels if l == cls)
        print(f"  {cls}: {count}")

    # ================= 特征提取（弱监督上下文：仅用于表示层）==================
    print("\n3.1 正在进行【增强特征提取（含天气）】...")
    print("📌 弱监督上下文：OSM/天气特征仅用于表示层，不影响决策层")
    feature_extractor = FeatureExtractorWithWeather(kg, weather_processor)
    all_features_and_labels = []

    success_count = 0
    degraded_count = 0

    for trajectory, datetime_series, label_str in tqdm(processed_segments_with_time,
                                                        desc="[Exp5 特征提取]"):
        try:
            trajectory_features, kg_features, weather_features = feature_extractor.extract_features(
                trajectory, datetime_series
            )

            assert trajectory_features.shape == (FIXED_SEQUENCE_LENGTH, TRAJECTORY_FEATURE_DIM)
            assert kg_features.shape == (FIXED_SEQUENCE_LENGTH, KG_FEATURE_DIM)
            assert weather_features.shape == (FIXED_SEQUENCE_LENGTH, WEATHER_FEATURE_DIM)

            label_encoded = label_encoder.transform([label_str])[0]
            all_features_and_labels.append((
                trajectory_features, kg_features, weather_features, label_encoded
            ))
            success_count += 1

        except Exception as e:
            try:
                trajectory_features = normalize_features_safe(trajectory)
                if trajectory_features.shape != (FIXED_SEQUENCE_LENGTH, TRAJECTORY_FEATURE_DIM):
                    if trajectory_features.shape[0] != FIXED_SEQUENCE_LENGTH:
                        if trajectory_features.shape[0] > FIXED_SEQUENCE_LENGTH:
                            trajectory_features = trajectory_features[:FIXED_SEQUENCE_LENGTH]
                        else:
                            padding = np.zeros((FIXED_SEQUENCE_LENGTH - trajectory_features.shape[0],
                                              TRAJECTORY_FEATURE_DIM), dtype=np.float32)
                            trajectory_features = np.vstack([trajectory_features, padding])
                    if trajectory_features.shape[1] != TRAJECTORY_FEATURE_DIM:
                        if trajectory_features.shape[1] > TRAJECTORY_FEATURE_DIM:
                            trajectory_features = trajectory_features[:, :TRAJECTORY_FEATURE_DIM]
                        else:
                            padding = np.zeros((FIXED_SEQUENCE_LENGTH,
                                              TRAJECTORY_FEATURE_DIM - trajectory_features.shape[1]),
                                              dtype=np.float32)
                            trajectory_features = np.hstack([trajectory_features, padding])

                kg_features = np.zeros((FIXED_SEQUENCE_LENGTH, KG_FEATURE_DIM), dtype=np.float32)
                weather_features = np.zeros((FIXED_SEQUENCE_LENGTH, WEATHER_FEATURE_DIM), dtype=np.float32)

                label_encoded = label_encoder.transform([label_str])[0]
                all_features_and_labels.append((
                    trajectory_features, kg_features, weather_features, label_encoded
                ))
                degraded_count += 1

            except Exception as inner_e:
                trajectory_features = np.zeros((FIXED_SEQUENCE_LENGTH, TRAJECTORY_FEATURE_DIM),
                                              dtype=np.float32)
                kg_features = np.zeros((FIXED_SEQUENCE_LENGTH, KG_FEATURE_DIM), dtype=np.float32)
                weather_features = np.zeros((FIXED_SEQUENCE_LENGTH, WEATHER_FEATURE_DIM), dtype=np.float32)

                try:
                    label_encoded = label_encoder.transform([label_str])[0]
                except:
                    label_encoded = 0

                all_features_and_labels.append((
                    trajectory_features, kg_features, weather_features, label_encoded
                ))
                degraded_count += 1

    print(f"\n✅ 特征提取完成:")
    print(f"   完整特征: {success_count}")
    print(f"   降级特征: {degraded_count}")
    print(f"   总样本数: {len(all_features_and_labels)}")

    # 保存缓存
    try:
        with open(cache_paths['features'], 'wb') as f:
            pickle.dump((all_features_and_labels, label_encoder, cleaning_stats), f,
                       protocol=pickle.HIGHEST_PROTOCOL)
        kg.save_cache(cache_paths['grid'])
        save_cache_metadata(cache_paths, osm_path, weather_path, geolife_root,
                           len(all_features_and_labels), label_encoder,
                           cleaning_stats, cleaning_mode)
    except Exception as e:
        print(f"⚠️ 缓存保存失败: {e}")

    return all_features_and_labels, kg, weather_processor, label_encoder, cleaning_stats


def train_epoch(model, dataloader, criterion, optimizer, device,
                max_grad_norm: float = 1.0, context_loss_weight: float = 0.1):
    """训练一个 epoch（支持弱监督上下文损失）

    Args:
        model: 弱监督上下文模型
        dataloader: 数据加载器
        criterion: 分类损失函数（交叉熵）
        optimizer: 优化器
        device: 设备
        max_grad_norm: 梯度裁剪阈值
        context_loss_weight: 上下文损失权重 λ

    Returns:
        avg_loss: 平均总损失
        accuracy: 训练准确率
    """
    model.train()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_context_loss = 0.0
    correct = 0
    total = 0
    nan_batches = 0

    for batch_idx, (traj_f, kg_f, weather_f, labels) in enumerate(
        tqdm(dataloader, desc="Training Progress", leave=True)
    ):
        try:
            traj_f = traj_f.to(device)
            kg_f = kg_f.to(device)
            weather_f = weather_f.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # 前向传播（获取轨迹表示和上下文表示）
            logits, trajectory_repr, context_repr = model(
                traj_f, kg_f, weather_f, return_context=True
            )

            # 计算分类损失（交叉熵）
            ce_loss = criterion(logits, labels)

            # 计算上下文一致性损失（embedding-level约束）
            context_loss = model.compute_context_loss(trajectory_repr, context_repr)

            # 总损失 = CE + λ * L_context
            total_batch_loss = ce_loss + context_loss_weight * context_loss

            # 反向传播
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

            total_loss += total_batch_loss.item()
            total_ce_loss += ce_loss.item()
            total_context_loss += context_loss.item()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        except RuntimeError as e:
            nan_batches += 1
            optimizer.zero_grad()
            continue

    if nan_batches > 0:
        print(f"   ⚠️ 本 epoch 跳过 {nan_batches} 个异常批次")

    num_batches = max(len(dataloader) - nan_batches, 1)
    avg_loss = total_loss / num_batches
    avg_ce_loss = total_ce_loss / num_batches
    avg_context_loss = total_context_loss / num_batches
    accuracy = correct / max(total, 1)

    return avg_loss, accuracy, avg_ce_loss, avg_context_loss


def evaluate(model, dataloader, criterion, device, label_encoder):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for traj_f, kg_f, weather_f, labels in tqdm(dataloader, desc="Validation Progress", leave=True):
            traj_f = traj_f.to(device)
            kg_f = kg_f.to(device)
            weather_f = weather_f.to(device)
            labels = labels.to(device)

            outputs = model(traj_f, kg_f, weather_f)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            total_loss += criterion(logits, labels).item()

            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    report = classification_report(
        all_labels, all_preds,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0
    )

    return total_loss / len(dataloader), report, all_preds, all_labels


def main():
    # ========================================================================
    # PyCharm 直接运行配置区域
    # ========================================================================
    PYCHARM_MODE = True

    if PYCHARM_MODE:
        class Args:
            geolife_root = '../data/Geolife Trajectories 1.3'
            osm_path = '../data/exp3.geojson'
            weather_path = '../data/beijing_weather_hourly_2007_2012.csv'

            use_base_data = True
            max_users = None

            # ✅ 新增：清洗模式配置
            cleaning_mode = 'balanced'  # 可选: strict, balanced, gentle

            batch_size = 32
            epochs = 50
            lr = 5e-5
            hidden_dim = 128
            num_layers = 2
            dropout = 0.3
            max_grad_norm = 1.0

            save_dir = 'checkpoints'
            num_workers = 0
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            clear_cache = False
            seed = 42

        args = Args()
        print(f"📌 使用 PyCharm 模式 (清洗模式: {args.cleaning_mode})")
    else:
        parser = argparse.ArgumentParser(description='训练交通方式识别模型 (Exp5 - 改进版)')
        parser.add_argument('--geolife_root', type=str, default='../data/Geolife Trajectories 1.3')
        parser.add_argument('--osm_path', type=str, default='../data/exp3.geojson')
        parser.add_argument('--weather_path', type=str, default='../data/beijing_weather_hourly_2007_2012.csv')

        parser.add_argument('--use_base_data', action='store_true', default=True)
        parser.add_argument('--max_users', type=int, default=None)

        # ✅ 新增：清洗模式参数
        parser.add_argument('--cleaning_mode', type=str, default='balanced',
                          choices=['strict', 'balanced', 'gentle'],
                          help='数据清洗模式: strict(严格), balanced(平衡), gentle(温和)')

        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--epochs', type=int, default=50)
        parser.add_argument('--lr', type=float, default=5e-5)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--num_layers', type=int, default=2)
        parser.add_argument('--dropout', type=float, default=0.3)
        parser.add_argument('--max_grad_norm', type=float, default=1.0)

        parser.add_argument('--save_dir', type=str, default='checkpoints')
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--device', type=str,
                          default='cuda' if torch.cuda.is_available() else 'cpu')
        parser.add_argument('--clear_cache', action='store_true')
        parser.add_argument('--seed', type=int, default=42)

        args = parser.parse_args()

    # ========================================================================

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("Exp5 交通方式识别训练 (改进版 - 数据清洗 + 弱监督上下文)")
    print("=" * 80)
    print(f"设备: {args.device}")
    print(f"清洗模式: {args.cleaning_mode}")
    print(f"学习率: {args.lr}")
    print(f"梯度裁剪: {args.max_grad_norm}")
    print(f"批次大小: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print("=" * 80 + "\n")

    # 清理缓存
    if args.clear_cache:
        cache_paths = get_cache_paths(args.cleaning_mode)
        print(f"正在清理缓存 (模式: {args.cleaning_mode})...")
        for key in ['kg', 'grid', 'weather', 'features', 'meta']:
            if os.path.exists(cache_paths[key]):
                os.remove(cache_paths[key])
                print(f"  已删除: {cache_paths[key]}")

    # 加载数据
    all_features_and_labels, kg, weather_processor, label_encoder, cleaning_stats = load_data(
        args.geolife_root, args.osm_path, args.weather_path, args.max_users,
        use_base_data=args.use_base_data, cleaning_mode=args.cleaning_mode
    )

    # 数据集划分
    num_classes = len(label_encoder.classes_)
    print(f"\n类别数量: {num_classes}")
    print(f"类别列表: {label_encoder.classes_}")

    dataset = TrajectoryDatasetWithWeather(all_features_and_labels)

    print(f"\n✅ 数据集大小:")
    print(f"  总样本数: {len(dataset)}")

    all_indices = np.arange(len(dataset))

    # 获取每个类别的标签
    labels_stratify = [label_encoder.inverse_transform([label_encoded])[0]
                       for _, _, _, label_encoded in all_features_and_labels]

    # 统计每个类别的样本数
    label_counts = Counter(labels_stratify)

    # 过滤掉样本数少于 2 的类别
    valid_labels = [label for label, count in label_counts.items() if count >= 2]

    # 只保留有效标签对应的样本索引
    valid_indices = [i for i, label in enumerate(labels_stratify) if label in valid_labels]
    valid_labels_stratify = [labels_stratify[i] for i in valid_indices]
    valid_all_features_and_labels = [all_features_and_labels[i] for i in valid_indices]

    # 执行第一次数据划分，使用有效标签对应的样本
    train_indices, temp_indices = train_test_split(
        valid_indices, test_size=0.3, random_state=42, stratify=valid_labels_stratify
    )

    # 在进行第二次划分时，确保 stratify 中的每个类别至少有 2 个样本
    temp_labels = [valid_labels_stratify[i] for i in temp_indices]
    # 过滤掉 temp_labels 中样本数少于 2 的类别
    temp_label_counts = Counter(temp_labels)
    valid_temp_labels = [label for label, count in temp_label_counts.items() if count >= 2]
    valid_temp_indices = [i for i, label in enumerate(temp_labels) if label in valid_temp_labels]
    valid_temp_labels_stratify = [temp_labels[i] for i in valid_temp_indices]

    # 执行第二次数据划分，确保 stratify 中的每个类别至少有 2 个样本
    val_indices, test_indices = train_test_split(
        valid_temp_indices, test_size=0.6667, random_state=42, stratify=valid_temp_labels_stratify
    )
    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, train_indices),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(dataset, val_indices),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    test_loader = DataLoader(
        torch.utils.data.Subset(dataset, test_indices),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    print(f"\n✅ 数据加载完成:")
    print(f"  Train: {len(train_indices)} 样本")
    print(f"  Val:   {len(val_indices)} 样本")
    print(f"  Test:  {len(test_indices)} 样本")

    # 构建弱监督上下文模型
    model = WeaklySupervisedContextModel(
        trajectory_feature_dim=TRAJECTORY_FEATURE_DIM,
        kg_feature_dim=KG_FEATURE_DIM,
        weather_feature_dim=WEATHER_FEATURE_DIM,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=num_classes,
        dropout=args.dropout,
        context_loss_type='mse',
        context_loss_weight=0.05
    ).to(args.device)

    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    criterion = nn.CrossEntropyLoss()

    # 上下文损失权重 λ
    context_loss_weight = 0.05

    patience = 10
    best_val_loss = float("inf")
    epochs_no_improve = 0

    print("\n" + "=" * 80)
    print("开始训练（弱监督上下文表示增强）")
    print("=" * 80)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # 训练（包含上下文一致性损失）
        train_loss, train_acc, train_ce_loss, train_context_loss = train_epoch(
            model, train_loader, criterion, optimizer, args.device,
            max_grad_norm=args.max_grad_norm,
            context_loss_weight=context_loss_weight
        )

        # 验证（仅使用分类损失）
        val_loss, report, _, _ = evaluate(
            model, val_loader, criterion, args.device, label_encoder
        )
        val_acc = report.get('accuracy', 0.0)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  CE Loss: {train_ce_loss:.4f} | Context Loss: {train_context_loss:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'label_encoder': label_encoder,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'model_config': {
                    'trajectory_feature_dim': TRAJECTORY_FEATURE_DIM,
                    'kg_feature_dim': KG_FEATURE_DIM,
                    'weather_feature_dim': WEATHER_FEATURE_DIM,
                    'hidden_dim': args.hidden_dim,
                    'num_layers': args.num_layers,
                    'num_classes': num_classes,
                    'dropout': args.dropout,
                    'context_loss_type': 'mse',
                    'context_loss_weight': context_loss_weight
                },
                'cleaning_stats': cleaning_stats,
                'cleaning_mode': args.cleaning_mode
            }

            model_path = os.path.join(args.save_dir, f'exp5_model_{args.cleaning_mode}.pth')
            torch.save(checkpoint, model_path)
            print(f"✓ 保存最佳模型: {model_path}")
        else:
            epochs_no_improve += 1
            print(f"⏳ 验证损失未改善: {epochs_no_improve}/{patience}")

            if epochs_no_improve >= patience:
                print(f"\n🛑 Early stopping 触发（patience={patience}）")
                break

    # 最终测试集评估
    print("\n" + "=" * 80)
    print("最终测试集评估")
    print("=" * 80)

    test_loss, test_report, _, _ = evaluate(
        model, test_loader, criterion, args.device, label_encoder
    )
    test_acc = test_report.get('accuracy', 0.0)

    print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    print("\n各类别详细指标:")
    for cls in label_encoder.classes_:
        if cls in test_report:
            metrics = test_report[cls]
            print(f"  {cls:15s}: P={metrics['precision']:.4f}, "
                  f"R={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}")

    # 训练完成
    print("\n" + "=" * 80)
    print("训练完成!")
    print("=" * 80)
    print(f"清洗模式: {args.cleaning_mode}")
    print(f"最佳验证 Loss: {best_val_loss:.4f}")
    print(f"最终测试 Accuracy: {test_acc:.4f}")

    model_path = os.path.join(args.save_dir, f'exp5_model_{args.cleaning_mode}.pth')
    print(f"模型已保存到: {model_path}")

    # 打印清洗统计
    if cleaning_stats:
        print("\n" + "=" * 80)
        print("数据清洗统计")
        print("=" * 80)

        before = cleaning_stats.get('before', {})
        after = cleaning_stats.get('after', {})
        quality = cleaning_stats.get('quality', {})
        cleaner = cleaning_stats.get('cleaner', {})

        print(f"\n第一阶段（基础处理）:")
        print(f"  总轨迹段数: {before.get('total_segments', 0):,}")
        print(f"  总轨迹点数: {before.get('total_points', 0):,}")

        print(f"\n第二阶段（深度清洗 - {args.cleaning_mode} 模式）:")
        print(f"  有效轨迹段数: {after.get('valid_segments', 0):,}")
        print(f"  丢弃轨迹段数: {after.get('total_discarded', 0):,}")
        print(f"  保留率: {after.get('retention_rate', 0):.2%}")

        print(f"\n清洗详情:")
        print(f"  剔除异常点数: {cleaner.get('outliers_removed', 0):,}")
        print(f"  插值点数: {cleaner.get('points_interpolated', 0):,}")
        print(f"  平滑点数: {cleaner.get('points_smoothed', 0):,}")

        # 新增：质量评估统计
        if quality:
            print(f"\n质量评估:")
            print(f"  平均质量分数: {quality.get('mean_quality', 0):.3f}")
            print(f"  高质量样本数: {quality.get('high_quality_count', 0):,} "
                  f"({quality.get('high_quality_ratio', 0):.1%})")
            print(f"  中等质量样本数: {quality.get('medium_quality_count', 0):,}")
            print(f"  低质量样本数: {quality.get('low_quality_count', 0):,}")


if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)
    main()