import pandas as pd
from analysis.technical.configs.config import TAConfig
from analysis.technical.indicators.indicator import TechnicalIndicator
from analysis.technical.indicators.dema import DEMA
from analysis.technical.indicators.vwap import VWAP
from analysis.technical.indicators.macd import MACDSignalDiffs
from analysis.technical.indicators.rsi import RSI
from analysis.technical.indicators.stoch import STOCH
from analysis.technical.indicators.cci import CCI
from analysis.technical.indicators.adx import ADX
from analysis.technical.indicators.aroons import AROONS
from analysis.technical.indicators.bbands import BBANDS
from analysis.technical.indicators.adl import ADL
from analysis.technical.indicators.obv import OBV


class StandardTAConfig(TAConfig):
    def __init__(
            self,
            dema_window: int = 15,
            vwap_window: int = 10,
            macd_short_window: int = 12,
            macd_long_window: int = 26,
            macd_signal_period: int = 9,
            rsi_window: int = 14,
            stoch_window: int = 14,
            cci_window: int = 20,
            adx_window: int = 14,
            aroons_window: int = 25,
            bbands_window: int = 20
    ):
        super().__init__()

        self._dema_window = dema_window
        self._vwap_window = vwap_window
        self._macd_short_window = macd_short_window
        self._macd_long_window = macd_long_window
        self._macd_signal_period = macd_signal_period
        self._rsi_window = rsi_window
        self._stoch_window = stoch_window
        self._cci_window = cci_window
        self._adx_window = adx_window
        self._aroons_window = aroons_window
        self._bbands_window = bbands_window

    def get_technical_analysis_config_dict(
            self,
            opens: pd.Series,
            highs: pd.Series,
            lows: pd.Series,
            closes: pd.Series,
            volumes: pd.Series
    ) -> dict[TechnicalIndicator, dict]:
        return {
            DEMA(): {'closes': closes, 'window': self._dema_window},
            VWAP(): {
                'highs': highs,
                'lows': lows,
                'closes': closes,
                'volumes': volumes,
                'window': self._vwap_window
            },
            MACDSignalDiffs(): {
                'closes': closes,
                'short_window': self._macd_short_window,
                'long_window': self._macd_long_window,
                'signal_period': self._macd_signal_period
            },
            RSI(): {'closes': closes, 'window': self._rsi_window},
            STOCH(): {'highs': highs, 'lows': lows, 'closes': closes, 'window': self._stoch_window},
            CCI(): {'highs': highs, 'lows': lows, 'closes': closes, 'window': self._cci_window},
            ADX(): {'highs': highs, 'lows': lows, 'closes': closes, 'window': self._adx_window},
            AROONS(): {'closes': closes, 'window': self._aroons_window},
            BBANDS(): {'closes': closes, 'window': self._bbands_window},
            ADL(): {'highs': highs, 'lows': lows, 'closes': closes, 'volumes': volumes},
            OBV(): {'closes': closes, 'volumes': volumes}
        }
