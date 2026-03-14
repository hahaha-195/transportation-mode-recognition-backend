# common/exp5_adapter.py
from common.exp4_adapter import Exp4DataAdapter


class Exp5DataAdapter(Exp4DataAdapter):
    """Exp5: 与Exp4格式相同"""

    @property
    def experiment_name(self):
        return "Exp5"
