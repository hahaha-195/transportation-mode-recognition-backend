# api_exp2.py
"""
Exp2 真实模型预测服务 (FastAPI)
提供 /predict 接口，接收轨迹点数据，返回预测的交通方式和置信度。
"""

import os
import sys
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

# 将项目根目录加入 Python 路径，以便导入 common、src 和 exp2 模块
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# 将 exp2 目录也加入路径，以便 predict.py 能找到 src
EXP2_PATH = os.path.join(PROJECT_ROOT, 'exp2')
if EXP2_PATH not in sys.path:
    sys.path.insert(0, EXP2_PATH)
# 尝试导入所需模块
try:
    # from common.feature_extraction import FeatureExtractor
    from exp2.predict import TransportationPredictorExp2
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确保以下文件存在：")
    print("  - common/feature_extraction.py")
    print("  - exp2/predict.py")
    print("  - src/model.py (被 predict.py 依赖)")
    sys.exit(1)

# ========== 虚拟知识图谱类（用于特征提取器） ==========
class DummyKnowledgeGraph:
    """当真实 KG 不可用时，返回全零 KG 特征矩阵 (N, 11)"""
    def extract_kg_features(self, trajectory: np.ndarray) -> np.ndarray:
        """
        返回与轨迹点数量相同的全零 KG 特征矩阵
        Args:
            trajectory: (N, 9) 轨迹特征矩阵（未使用）
        Returns:
            (N, 11) 全零矩阵
        """
        n_points = trajectory.shape[0]
        return np.zeros((n_points, 11), dtype=np.float32)

# ========== 初始化 ==========
# 模型检查点路径
MODEL_PATH = os.path.join(PROJECT_ROOT, "exp2", "checkpoints", "exp2_model.pth")

# 加载模型
try:
    predictor = TransportationPredictorExp2(checkpoint_path=MODEL_PATH)
    print(f"✅ 模型加载成功，支持类别: {predictor.class_names}")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    predictor = None

# 初始化特征提取器（使用虚拟 KG）
# dummy_kg = DummyKnowledgeGraph()
# feature_extractor = FeatureExtractor(kg=dummy_kg)

# ========== FastAPI 应用 ==========
app = FastAPI(title="Exp2 交通方式识别 API", description="基于多源数据融合的交通方式识别系统 - 实验二")

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境请替换为具体前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== 数据模型 ==========
class TrajectoryPoint(BaseModel):
    lat: float
    lng: float
    timestamp: int  # 秒级时间戳

class PredictRequest(BaseModel):
    trajectory: List[TrajectoryPoint]

class PredictResponse(BaseModel):
    prediction: str
    confidence: float

# ========== 工具函数 ==========
def points_to_numpy(points: List[TrajectoryPoint]) -> np.ndarray:
    """
    将前端轨迹点列表转换为 NumPy 数组，格式符合 FeatureExtractor 要求：
        [:, 0] = latitude
        [:, 1] = longitude
        [:, 2] = speed (暂填0，将由特征提取器计算)
        [:, 3] = acceleration
        [:, 4] = bearing_change
        [:, 5] = distance
        [:, 6] = time_diff
        [:, 7] = total_distance
        [:, 8] = total_time
    注意：FeatureExtractor 的 _extract_trajectory_features 会重新计算这些特征，
          但我们仍需传递经纬度和时间信息。这里我们只填充经纬度和时间戳，
          其余特征由特征提取器内部计算。
    """
    n = len(points)
    # 初始化数组，除了经纬度和时间戳外，其余列暂时填0
    arr = np.zeros((n, 9), dtype=np.float32)
    for i, p in enumerate(points):
        arr[i, 0] = p.lat
        arr[i, 1] = p.lng
        # 将时间戳转换为相对时间（秒），第一个点为0
        if i == 0:
            arr[i, 6] = 0  # time_diff
            base_time = p.timestamp
        else:
            arr[i, 6] = p.timestamp - points[i-1].timestamp
    # 注意：这里没有填充速度、加速度等，这些将由特征提取器内部计算。
    # 但 FeatureExtractor._extract_trajectory_features 期望输入已经包含这些特征吗？
    # 实际上，_extract_trajectory_features 只是对已有的9维特征进行归一化，并不重新计算。
    # 我们需要重新设计：要么自己计算所有9维特征，要么修改特征提取器。
    # 这里我们选择自己计算所有9维特征，使用文档中的公式。
    # 因此，我们重写一个函数来计算完整的9维特征。
    raise NotImplementedError("需要实现完整的轨迹特征计算函数")

# 由于 FeatureExtractor 的 _extract_trajectory_features 仅做归一化，不计算特征，
# 我们需要自己实现特征计算。这里从《交通识别系统.md》5.3节提取函数。
def extract_trajectory_features(points: List[TrajectoryPoint]) -> np.ndarray:
    """
    从GPS点列表提取9维轨迹特征
    输入: points - 包含 lat, lng, timestamp 的对象列表
    输出: numpy array, shape (n_points, 9)
    """
    n = len(points)
    features = np.zeros((n, 9), dtype=np.float32)
    if n == 0:
        return features

    # 第一个点
    features[0, 0] = points[0].lat
    features[0, 1] = points[0].lng
    # 其余特征在后续点计算

    total_distance = 0.0
    total_time = 0.0
    prev_lat, prev_lon = points[0].lat, points[0].lng
    prev_time = points[0].timestamp
    prev_speed = 0.0
    prev_bearing = 0.0

    for i in range(1, n):
        lat, lon = points[i].lat, points[i].lng
        timestamp = points[i].timestamp

        # 距离和时间差
        dist = haversine_distance(prev_lat, prev_lon, lat, lon)
        time_diff = timestamp - prev_time
        if time_diff <= 0:
            time_diff = 0.1  # 避免除零

        # 速度
        speed = dist / time_diff

        # 加速度
        acceleration = (speed - prev_speed) / time_diff

        # 方向变化
        bearing = calculate_bearing(prev_lat, prev_lon, lat, lon)
        bearing_change = abs(bearing - prev_bearing)
        if bearing_change > 180:
            bearing_change = 360 - bearing_change

        # 累计值
        total_distance += dist
        total_time += time_diff

        # 填充特征
        features[i, 0] = lat
        features[i, 1] = lon
        features[i, 2] = speed
        features[i, 3] = acceleration
        features[i, 4] = bearing_change
        features[i, 5] = dist
        features[i, 6] = time_diff
        features[i, 7] = total_distance
        features[i, 8] = total_time

        # 更新前值
        prev_lat, prev_lon = lat, lon
        prev_time = timestamp
        prev_speed = speed
        prev_bearing = bearing

    # 对特征进行归一化（使用与 FeatureExtractor 相同的 Z-score）
    features = normalize_features(features)
    return features

def haversine_distance(lat1, lon1, lat2, lon2):
    """计算两点间的距离（米）"""
    R = 6371000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))

def calculate_bearing(lat1, lon1, lat2, lon2):
    """计算航向角（度）"""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.arctan2(x, y)
    return (np.degrees(bearing) + 360) % 360

def normalize_features(features: np.ndarray) -> np.ndarray:
    """Z-score 归一化，截断到 [-5,5]"""
    mean = np.mean(features, axis=0, keepdims=True)
    std = np.std(features, axis=0, keepdims=True) + 1e-8
    normalized = (features - mean) / std
    return np.clip(normalized, -5, 5)

def generate_kg_features(trajectory_points: List[TrajectoryPoint]) -> np.ndarray:
    """
    生成知识图谱特征向量 (11维)
    当前版本：全零向量（占位），可根据实际需求替换为真实 KG 特征提取逻辑。
    """
    # TODO: 替换为真实的 KG 特征提取（例如从 OSM 数据中计算）
    return np.zeros(11, dtype=np.float32)

# ========== API 端点 ==========
@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if predictor is None:
        raise HTTPException(status_code=500, detail="模型未加载，请检查服务端日志")

    try:
        # 1. 提取轨迹特征 (seq_len, 9)
        traj_features = extract_trajectory_features(request.trajectory)
        if traj_features.shape[1] != 9:
            raise ValueError(f"轨迹特征维度错误：预期 9 维，实际 {traj_features.shape[1]} 维")

        # 2. 生成知识图谱特征 (11,)
        kg_features = generate_kg_features(request.trajectory)

        # 3. 调用模型预测
        pred_label, confidence = predictor.predict(traj_features, kg_features)

        return PredictResponse(prediction=pred_label, confidence=float(confidence))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "ok" if predictor else "degraded",
        "model_loaded": predictor is not None,
        "classes": predictor.class_names.tolist() if predictor else []
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)