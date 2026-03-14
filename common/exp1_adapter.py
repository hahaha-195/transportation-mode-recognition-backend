# common/exp1_adapter.py
from common.base_adapter import BaseDataAdapter
from typing import List, Tuple
import numpy as np


class Exp1DataAdapter(BaseDataAdapter):
    """Exp1: 返回 (features, label)"""

    def __init__(self, enable_cleaning=True, cleaning_mode='balanced',
                 cache_dir='../data/processed'):
        super().__init__(enable_cleaning, cleaning_mode, cache_dir)

    @property
    def experiment_name(self):
        return "Exp1"

    def _format_output(self, cleaned_segments: List[Tuple]) -> List[Tuple[np.ndarray, str]]:
        """返回 (features, label)，丢弃时间序列"""
        return [(traj, label) for traj, _, label in cleaned_segments]