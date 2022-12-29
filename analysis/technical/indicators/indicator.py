import pandas as pd
from abc import ABC, abstractmethod


class TechnicalIndicator(ABC):
    def __init__(self, name: str or tuple[str, str]):
        self._name = name

    @property
    def name(self) -> str or tuple[str, str]:
        return self._name

    def __hash__(self) -> int:
        return hash(self.name)

    def __call__(self, **kwargs) -> pd.Series or tuple[pd.Series, pd.Series]:
        return self.compute_indicator_values(**kwargs)

    def compute_indicator_values(self, **kwargs) -> pd.Series or tuple[pd.Series, pd.Series]:
        indicator_values = self._compute_indicator_values(**kwargs)

        assert (isinstance(self.name, str) and isinstance(indicator_values, pd.Series)) or \
               (isinstance(self.name, tuple) and isinstance(indicator_values, tuple)), \
            'AssertionError Indicator names does not match indicator values'

        return indicator_values

    @abstractmethod
    def _compute_indicator_values(self, **kwargs) -> pd.Series or tuple[pd.Series, pd.Series]:
        pass
