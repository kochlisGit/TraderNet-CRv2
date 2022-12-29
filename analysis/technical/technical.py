import pandas as pd
from analysis.technical.indicators.indicator import TechnicalIndicator


class TechnicalAnalysis:
    def __init__(self, dates: pd.Series):
        self._dates = dates

    def compute_technical_indicators(self, ta_config_dict: dict[TechnicalIndicator, dict]) -> pd.DataFrame:
        ta = {'date': self._dates}

        for indicator, params in ta_config_dict.items():
            indicator_name = indicator.name
            indicator_values = indicator.compute_indicator_values(**params)

            if isinstance(indicator_name, str):
                ta[indicator_name] = indicator_values
            else:
                for name, values in zip(indicator_name, indicator_values):
                    ta[name] = values
        return pd.DataFrame(data=ta)
