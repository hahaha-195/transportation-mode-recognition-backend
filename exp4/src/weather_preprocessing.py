"""
天气数据预处理模块 (Exp4 - 稳定版)
功能：加载、处理和特征化天气数据
增强：全面的缺失值处理，确保无 NaN/Inf 输出
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
import warnings


class WeatherDataProcessor:
    """天气数据处理器 (稳定版 - 支持大量缺失数据)"""

    # 天气特征维度常量
    WEATHER_FEATURE_DIM = 12

    def __init__(self, weather_csv_path: str):
        """
        初始化天气数据处理器

        Args:
            weather_csv_path: 天气CSV文件路径
        """
        self.weather_csv_path = weather_csv_path
        self.daily_weather = None
        self.weather_features_cache = {}
        self._default_features = np.zeros(self.WEATHER_FEATURE_DIM, dtype=np.float32)
        self._load_successful = False

    def load_and_process(self) -> pd.DataFrame:
        """
        加载并处理天气数据（带完整异常处理）

        Returns:
            daily_weather: 日级别聚合的天气数据（可能为空 DataFrame）
        """
        print("\n========== 天气数据加载与处理 ==========")

        try:
            # 1. 加载小时级数据
            print(f"1. 正在加载天气数据: {self.weather_csv_path}")
            hourly_data = pd.read_csv(self.weather_csv_path, index_col=0, parse_dates=True)
            print(f"   加载完成: {len(hourly_data)} 条小时记录")

            if len(hourly_data) == 0:
                print("   ⚠️ 天气数据为空，将使用默认零向量")
                self._initialize_empty_weather()
                return self.daily_weather

            # 2. 数据清洗
            print("2. 正在清洗数据...")
            hourly_data = self._clean_hourly_data(hourly_data)

            # 3. 聚合为日级数据
            print("3. 正在聚合为日级数据...")
            self.daily_weather = self._aggregate_to_daily(hourly_data)
            print(f"   聚合完成: {len(self.daily_weather)} 条日记录")

            # 4. 构造天气特征
            print("4. 正在构造天气特征...")
            self.daily_weather = self._construct_weather_features(self.daily_weather)

            # 5. 最终清理：确保无 NaN/Inf
            self.daily_weather = self._final_cleanup(self.daily_weather)

            self._load_successful = True
            print("✅ 天气数据处理完成")
            print(f"   日期范围: {self.daily_weather.index.min()} 至 {self.daily_weather.index.max()}")
            print(f"   特征列: {list(self.daily_weather.columns)}")

        except FileNotFoundError:
            print(f"   ⚠️ 天气文件不存在: {self.weather_csv_path}")
            self._initialize_empty_weather()
        except Exception as e:
            print(f"   ⚠️ 天气数据加载失败: {e}")
            self._initialize_empty_weather()

        return self.daily_weather

    def _initialize_empty_weather(self):
        """初始化空天气数据框架"""
        self.daily_weather = pd.DataFrame(columns=[
            'temp', 'prcp', 'snow', 'wspd', 'rhum',
            'is_rainy', 'is_heavy_rain', 'is_snowy',
            'is_cold', 'is_hot', 'is_windy'
        ])
        self._load_successful = False
        print("   使用空天气数据（所有特征将为零）")

    def _clean_hourly_data(self, hourly_data: pd.DataFrame) -> pd.DataFrame:
        """清洗小时级数据"""
        # 将 '<NA>' 字符串替换为 NaN
        hourly_data = hourly_data.replace('<NA>', np.nan)
        hourly_data = hourly_data.replace('', np.nan)
        hourly_data = hourly_data.replace('NA', np.nan)

        # 确保数值列为 float 类型
        numeric_columns = ['temp', 'dwpt', 'rhum', 'prcp', 'snow', 'wdir',
                           'wspd', 'wpgt', 'pres', 'tsun']
        for col in numeric_columns:
            if col in hourly_data.columns:
                hourly_data[col] = pd.to_numeric(hourly_data[col], errors='coerce')

        return hourly_data

    def _aggregate_to_daily(self, hourly_data: pd.DataFrame) -> pd.DataFrame:
        """
        将小时数据聚合为日数据

        聚合规则:
        - temp: 平均温度
        - prcp: 总降水量
        - snow: 总降雪量
        - wspd: 平均风速
        - rhum: 平均相对湿度
        """
        # 确保需要的列存在，不存在则创建空列
        required_cols = ['temp', 'prcp', 'snow', 'wspd', 'rhum']
        for col in required_cols:
            if col not in hourly_data.columns:
                hourly_data[col] = np.nan

        # 按日期分组
        try:
            daily_agg = hourly_data.groupby(hourly_data.index.date).agg({
                'temp': 'mean',  # 平均温度
                'prcp': 'sum',   # 总降水量
                'snow': 'sum',   # 总降雪量
                'wspd': 'mean',  # 平均风速
                'rhum': 'mean',  # 平均湿度
            })
        except Exception as e:
            print(f"   ⚠️ 聚合失败: {e}，使用空数据")
            return pd.DataFrame(columns=['temp', 'prcp', 'snow', 'wspd', 'rhum'])

        # 将索引转换为 datetime
        daily_agg.index = pd.to_datetime(daily_agg.index)

        # 填充缺失值（前向填充，然后后向填充，最后用默认值）
        daily_agg = daily_agg.ffill()
        daily_agg = daily_agg.bfill()

        # 剩余的NaN用合理默认值填充
        default_values = {
            'temp': 15.0,    # 默认温度 15°C
            'prcp': 0.0,     # 默认无降水
            'snow': 0.0,     # 默认无降雪
            'wspd': 3.0,     # 默认风速 3 m/s
            'rhum': 50.0,    # 默认湿度 50%
        }
        daily_agg = daily_agg.fillna(default_values)

        return daily_agg

    def _construct_weather_features(self, daily_data: pd.DataFrame) -> pd.DataFrame:
        """
        构造二值天气特征

        特征:
        - is_rainy: 是否有降水 (prcp > 0)
        - is_heavy_rain: 是否大雨 (prcp > 10)
        - is_snowy: 是否下雪 (snow > 0)
        - is_cold: 是否寒冷 (temp < 0)
        - is_hot: 是否炎热 (temp > 30)
        - is_windy: 是否风大 (wspd > 6)
        """
        df = daily_data.copy()

        # 确保基础列存在
        for col in ['prcp', 'snow', 'temp', 'wspd']:
            if col not in df.columns:
                df[col] = 0.0

        # 二值特征（安全处理 NaN）
        df['is_rainy'] = (df['prcp'].fillna(0) > 0).astype(int)
        df['is_heavy_rain'] = (df['prcp'].fillna(0) > 10).astype(int)
        df['is_snowy'] = (df['snow'].fillna(0) > 0).astype(int)
        df['is_cold'] = (df['temp'].fillna(15) < 0).astype(int)
        df['is_hot'] = (df['temp'].fillna(15) > 30).astype(int)
        df['is_windy'] = (df['wspd'].fillna(3) > 6).astype(int)

        return df

    def _final_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        """最终清理：确保无 NaN/Inf"""
        # 替换所有 NaN 和 Inf
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        return df

    def get_weather_features_for_date(self, date) -> np.ndarray:
        """
        获取指定日期的天气特征 (12维) - 稳定版

        Args:
            date: 日期时间戳（支持多种格式）

        Returns:
            weather_features: (12,) 天气特征向量，保证无 NaN/Inf
        """
        # 处理无效输入
        if date is None or pd.isna(date):
            return self._default_features.copy()

        # 提取日期部分
        try:
            if isinstance(date, pd.Timestamp):
                date_only = pd.Timestamp(date.date())
            elif isinstance(date, str):
                date_only = pd.Timestamp(pd.to_datetime(date).date())
            else:
                date_only = pd.Timestamp(pd.to_datetime(date).date())
        except Exception:
            return self._default_features.copy()

        # 检查缓存
        if date_only in self.weather_features_cache:
            return self.weather_features_cache[date_only].copy()

        # 如果天气数据未成功加载，返回默认值
        if not self._load_successful or self.daily_weather is None or len(self.daily_weather) == 0:
            features = self._default_features.copy()
            self.weather_features_cache[date_only] = features
            return features

        # 如果没有这一天的数据，尝试查找最近的
        if date_only not in self.daily_weather.index:
            try:
                # 找最近的日期
                time_diffs = abs(self.daily_weather.index - date_only)
                closest_idx = time_diffs.argmin()
                closest_date = self.daily_weather.index[closest_idx]

                # 如果超过7天，返回默认值
                if abs((closest_date - date_only).days) > 7:
                    features = self._default_features.copy()
                    self.weather_features_cache[date_only] = features
                    return features

                date_only = closest_date
            except Exception:
                features = self._default_features.copy()
                self.weather_features_cache[date_only] = features
                return features

        # 获取天气数据
        try:
            weather = self.daily_weather.loc[date_only]

            # 安全获取值
            temp = self._safe_get(weather, 'temp', 15.0)
            prcp = self._safe_get(weather, 'prcp', 0.0)
            snow = self._safe_get(weather, 'snow', 0.0)
            wspd = self._safe_get(weather, 'wspd', 3.0)
            rhum = self._safe_get(weather, 'rhum', 50.0)
            is_rainy = self._safe_get(weather, 'is_rainy', 0)
            is_heavy_rain = self._safe_get(weather, 'is_heavy_rain', 0)
            is_snowy = self._safe_get(weather, 'is_snowy', 0)
            is_cold = self._safe_get(weather, 'is_cold', 0)
            is_hot = self._safe_get(weather, 'is_hot', 0)
            is_windy = self._safe_get(weather, 'is_windy', 0)

            # 构造特征向量 (12维)
            features = np.array([
                temp,              # 0: 温度
                prcp,              # 1: 降水量
                snow,              # 2: 降雪量
                wspd,              # 3: 风速
                rhum,              # 4: 湿度
                is_rainy,          # 5: 是否降水
                is_heavy_rain,     # 6: 是否大雨
                is_snowy,          # 7: 是否下雪
                is_cold,           # 8: 是否寒冷
                is_hot,            # 9: 是否炎热
                is_windy,          # 10: 是否风大
                (temp + 20) / 50   # 11: 归一化温度 (假设范围 -20~30)
            ], dtype=np.float32)

            # 最终安全检查
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            features = np.clip(features, -100, 100)  # 合理范围裁剪

        except Exception:
            features = self._default_features.copy()

        # 缓存
        self.weather_features_cache[date_only] = features
        return features.copy()

    def _safe_get(self, series, key: str, default) -> float:
        """安全获取值，处理 NaN/Inf"""
        try:
            val = series[key] if key in series.index else default
            if pd.isna(val) or np.isinf(val):
                return float(default)
            return float(val)
        except Exception:
            return float(default)

    def get_weather_features_for_trajectory(self, trajectory_dates: pd.Series) -> np.ndarray:
        """
        获取轨迹的天气特征 (批量) - 稳定版

        Args:
            trajectory_dates: 轨迹的日期时间序列

        Returns:
            weather_features: (N, 12) 天气特征矩阵，保证无 NaN/Inf
        """
        # 处理空输入
        if trajectory_dates is None or len(trajectory_dates) == 0:
            return np.zeros((0, self.WEATHER_FEATURE_DIM), dtype=np.float32)

        N = len(trajectory_dates)
        weather_features = np.zeros((N, self.WEATHER_FEATURE_DIM), dtype=np.float32)

        for i, date in enumerate(trajectory_dates):
            try:
                weather_features[i] = self.get_weather_features_for_date(date)
            except Exception:
                # 任何异常都使用默认零向量
                weather_features[i] = self._default_features.copy()

        # 最终安全检查
        weather_features = np.nan_to_num(weather_features, nan=0.0, posinf=0.0, neginf=0.0)

        return weather_features

    def get_statistics(self) -> Dict:
        """获取天气数据统计信息"""
        if self.daily_weather is None or len(self.daily_weather) == 0:
            return {
                'total_days': 0,
                'date_range': 'N/A',
                'load_successful': self._load_successful
            }

        stats = {
            'total_days': len(self.daily_weather),
            'date_range': f"{self.daily_weather.index.min()} ~ {self.daily_weather.index.max()}",
            'avg_temp': float(self.daily_weather['temp'].mean()) if 'temp' in self.daily_weather else 0,
            'avg_prcp': float(self.daily_weather['prcp'].mean()) if 'prcp' in self.daily_weather else 0,
            'rainy_days': int(self.daily_weather['is_rainy'].sum()) if 'is_rainy' in self.daily_weather else 0,
            'snowy_days': int(self.daily_weather['is_snowy'].sum()) if 'is_snowy' in self.daily_weather else 0,
            'cold_days': int(self.daily_weather['is_cold'].sum()) if 'is_cold' in self.daily_weather else 0,
            'hot_days': int(self.daily_weather['is_hot'].sum()) if 'is_hot' in self.daily_weather else 0,
            'load_successful': self._load_successful
        }

        return stats