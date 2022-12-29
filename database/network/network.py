import pandas as pd
from database.entities.crypto import Crypto
from abc import ABC, abstractmethod


class DatasetDownloader(ABC):
    def __init__(self, date_column_name: str, verbose: bool):
        self._date_column_name = date_column_name
        self._verbose = verbose

    @property
    def date_column_name(self) -> str:
        return self._date_column_name

    @property
    def verbose(self) -> bool:
        return self._verbose

    def _store_dataset(self, dataset_df: pd.DataFrame, filepath: str, columns: list or None = None):
        assert not dataset_df.duplicated(subset=self.date_column_name).any(), \
            f'AssertionError: Date column is expected to be unique, got duplicates'

        assert dataset_df[self.date_column_name].is_monotonic_increasing, \
            f'AssertionError: Date column is expected to be monotonic and increasing'

        dataset_df.to_csv(filepath, columns=columns, index=False)

    @abstractmethod
    def download_historical_data(self, crypto: Crypto, history_filepath: str) -> bool:
        pass

    @abstractmethod
    def update_historical_data(self, crypto: Crypto, history_filepath: str) -> bool:
        pass
