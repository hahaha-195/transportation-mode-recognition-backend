# common/exp4_adapter.py
from common.base_adapter import BaseDataAdapter
from typing import List, Tuple
import numpy as np
import pandas as pd


class Exp4DataAdapter(BaseDataAdapter):
    """Exp4: 返回 (features, datetime_series, label)"""

    def __init__(self, enable_cleaning=True, cleaning_mode='balanced',
                 cache_dir='../data/processed'):
        super().__init__(enable_cleaning, cleaning_mode, cache_dir)

    @property
    def experiment_name(self):
        return "Exp4"

    def _format_output(self, cleaned_segments: List[Tuple]) -> List[Tuple[np.ndarray, pd.Series, str]]:
        """保留时间序列"""
        return cleaned_segments