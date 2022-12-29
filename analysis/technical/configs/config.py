import pandas as pd
from abc import ABC, abstractmethod
from analysis.technical.indicators.indicator import TechnicalIndicator


class TAConfig(ABC):
    @abstractmethod
    def get_technical_analysis_config_dict(
            self,
            opens: pd.Series,
            highs: pd.Series,
            lows: pd.Series,
            closes: pd.Series,
            volumes: pd.Series
    ) -> dict[TechnicalIndicator, dict]:
        pass
