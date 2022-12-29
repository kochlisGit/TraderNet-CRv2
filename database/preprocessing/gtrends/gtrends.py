import warnings
import numpy as np
import pandas as pd
from database.preprocessing.preprocessing import DatasetPreprocessing


class GoogleTrendsPreprocessing(DatasetPreprocessing):
    def __init__(
            self,
            impute_missing_scores: bool = False,
            imputing_percentage_threshold: float = 0.1
    ):
        assert 0.0 < imputing_percentage_threshold < 1.0, \
            'AssertionError: imputing_percentage_threshold is expected to be a float value between [0.0, 1.0], ' \
            f'got: {imputing_percentage_threshold}'

        self._impute_missing_scores = impute_missing_scores
        self._imputing_percentage_threshold = imputing_percentage_threshold

    def _impute_missing_trend_scores(self, trend_scores: pd.Series) -> pd.Series:
        missing_scores_percentage = trend_scores.isna().mean()

        if missing_scores_percentage < self._imputing_percentage_threshold:
            trend_scores[trend_scores == 0] = np.nan
            trend_scores.interpolate(method='polynomial', order=5, inplace=True)
            trend_scores[trend_scores == np.nan] = 0
        else:
            warnings.warn('Expected missing scores percentage to be less than '
                          f'{self._imputing_percentage_threshold}, got {missing_scores_percentage}%. '
                          f'Imputation process is skipped')
        return trend_scores

    def preprocess(self, trends_df: pd.DataFrame) -> pd.DataFrame:
        trends_df.rename(columns={trends_df.columns[1]: 'trends'}, inplace=True)

        if self._impute_missing_scores:
            trends_df['trends'] = self._impute_missing_trend_scores(trend_scores=trends_df['trends'])
        return trends_df
