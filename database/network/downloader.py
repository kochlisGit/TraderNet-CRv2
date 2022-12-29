from database.entities.crypto import Crypto
from database.network.coinapi.ohlcv import OHLCVDownloader
from database.network.gtrends.gtrends import GoogleTrendsDownloader


class CryptoDatasetDownloader:
    def download_crypto_datasets(
            self,
            crypto: Crypto,
            ohlcv_history_filepath: str,
            gtrends_history_filepath: str,
            historical_frequency: OHLCVDownloader.HistoricalFrequency or str,
            verbose: bool = True
    ) -> bool:
        ohlcv_downloader = OHLCVDownloader(historical_frequency=historical_frequency, verbose=verbose)
        gtrends_downloader = GoogleTrendsDownloader(verbose=verbose)

        if ohlcv_downloader.download_historical_data(
                crypto=crypto,
                history_filepath=ohlcv_history_filepath.format(crypto.symbol)
        ) and gtrends_downloader.download_historical_data(
            crypto=crypto,
            history_filepath=gtrends_history_filepath.format(crypto.symbol)
        ):
            if verbose:
                print(f'Successfully downloaded {crypto.symbol} dataset')

            return True

        if verbose:
            print('Download failed')

        return False
