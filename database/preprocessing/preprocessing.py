import pandas as pd
from abc import ABC, abstractmethod


class DatasetPreprocessing(ABC):
    @abstractmethod
    def preprocess(self, dataset_df: pd.DataFrame) -> pd.DataFrame:
        pass
