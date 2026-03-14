# common/exp2_adapter.py
from common.base_adapter import BaseDataAdapter
from typing import List, Tuple
import numpy as np


class Exp2DataAdapter(BaseDataAdapter):
    """Exp2: 返回 (features, label)"""

    def __init__(self, enable_cleaning=True, cleaning_mode='balanced',
                 cache_dir='../data/processed'):
        super().__init__(enable_cleaning, cleaning_mode, cache_dir)

    @property
    def experiment_name(self):
        return "Exp2"

    def _format_output(self, cleaned_segments: List[Tuple]) -> List[Tuple[np.ndarray, str]]:
        return [(traj, label) for traj, _, label in cleaned_segments]