# common/exp3_adapter.py
from common.exp2_adapter import Exp2DataAdapter


class Exp3DataAdapter(Exp2DataAdapter):
    """Exp3: 与Exp2格式相同"""

    @property
    def experiment_name(self):
        return "Exp3"