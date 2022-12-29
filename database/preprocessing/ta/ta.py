import pandas as pd
from database.preprocessing.preprocessing import DatasetPreprocessing


class TechnicalAnalysisPreprocessing(DatasetPreprocessing):
    def __init__(self, closes: pd.Series):
        self._closes = closes

    def preprocess(self, ta_df: pd.DataFrame) -> pd.DataFrame:
        ta_df_columns = set(ta_df.columns)

        if 'dema' in ta_df_columns:
            ta_df['close_dema'] = self._closes - ta_df['dema']
        if 'vwap' in ta_df_columns:
            ta_df['close_vwap'] = self._closes - ta_df['vwap']
        if 'bband_up' and 'bband_down' in ta_df_columns:
            ta_df['bband_up_close'] = ta_df['bband_up'] - self._closes
            ta_df['close_bband_down'] = self._closes - ta_df['bband_down']
        if 'adl' in ta_df_columns:
            ta_df['adl_diffs'] = ta_df['adl'].diff()
        if 'obv' in ta_df_columns:
            ta_df['obv_diffs'] = ta_df['obv'].diff()
        return ta_df
