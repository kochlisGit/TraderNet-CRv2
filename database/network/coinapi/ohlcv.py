import io
import pandas as pd
from enum import Enum
from database.entities.crypto import Crypto
from database.network.coinapi.coinapi import CoinAPIDownloader


class OHLCVDownloader(CoinAPIDownloader):
    class HistoricalFrequency(Enum):
        MINUTE = '1MIN'
        HOUR = '1HRS'

    def __init__(self, historical_frequency: HistoricalFrequency or str, verbose: bool):
        if isinstance(historical_frequency, str):
            if historical_frequency == '1HRS':
                self._historical_frequency = self.HistoricalFrequency.HOUR
            elif historical_frequency == '1MIN':
                self._historical_frequency = self.HistoricalFrequency.MINUTE
            else:
                raise NotImplementedError(f'"{historical_frequency}" frequency has not been implemented yet')
        else:
            self._historical_frequency = historical_frequency

        super().__init__(verbose=verbose)

        self._history_request_url = 'https://rest.coinapi.io/v1/ohlcv/{}/USD/history'
        self._latest_request_url = 'https://rest.coinapi.io/v1/ohlcv/{}/USD/latest'
        self._download_limit = 100000
        self._update_limit = 1000

    def _get_date_column_name(self) -> str:
        return 'time_period_end'

    def _get_request_params(self) -> dict:
        return {
            'period_id': self._historical_frequency.value,
            'output_format': 'csv',
            'csv_set_delimiter': ',',
            'time_start': '{}-{}-{}T00:00:00',
            'limit': '{}'
        }

    def download_historical_data(self, crypto: Crypto, history_filepath: str) -> bool:
        if self.verbose:
            print(f'Downloading {crypto.name} market history data for {crypto.start_year}')

        request_params = self._get_request_params()
        request_params['time_start'] = request_params['time_start'].format(crypto.start_year, '01', '01')
        request_params['limit'] = request_params['limit'].format(self._download_limit)
        base_url = self._history_request_url.format(crypto.symbol)

        response = self._get_response(
            base_url=base_url,
            request_params=request_params
        )
        if response is not None and response.status_code == 200:
            ohlcv_df = pd.read_csv(io.StringIO(response.text), sep=',')
            super()._store_dataset(dataset_df=ohlcv_df, filepath=history_filepath)
            return True
        else:
            return False

    def update_historical_data(self, crypto: Crypto, history_filepath: str) -> bool:
        if self.verbose:
            print(f'Updating {crypto.name} market latest data for {crypto.start_year}')

        request_params = self._get_request_params()
        request_params['limit'] = request_params['limit'].format(self._update_limit)
        del request_params['time_start']
        base_url = self._latest_request_url.format(crypto.symbol)

        response = self._get_response(
            base_url=base_url,
            request_params=request_params
        )

        if response is not None and response.status_code == 200:
            history_df = pd.read_csv(history_filepath)
            latest_df = pd.read_csv(io.StringIO(response.text), sep=',').sort_values(
                by=self.date_column_name, ascending=True
            )
            merged_df = pd.concat((history_df, latest_df), ignore_index=True)
            merged_df.drop_duplicates(subset=self.date_column_name, inplace=True)

            super()._store_dataset(dataset_df=merged_df, filepath=history_filepath)
            return True
        return False
