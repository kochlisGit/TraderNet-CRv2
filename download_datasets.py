import warnings
import config
from analysis.technical.configs.standard import StandardTAConfig
from database.datasets.builder import DatasetBuilder
from database.network.downloader import CryptoDatasetDownloader

warnings.filterwarnings('ignore')

ta_config = StandardTAConfig()
import_gtrends = True
impute_missing_gtrends = True
gtrends_imputing_percentage_threshold = 0.1
verbose = True


def main():
    downloader = CryptoDatasetDownloader()
    builder = DatasetBuilder()

    for crypto_symbol, crypto in config.supported_cryptos.items():
        if downloader.download_crypto_datasets(
            crypto=crypto,
            ohlcv_history_filepath=config.ohlcv_history_filepath,
            gtrends_history_filepath=config.gtrends_history_filepath,
            historical_frequency=config.ohlcv_dataset_period_id,
            verbose=verbose
        ):
            builder.build_dataset(
                ohlcv_history_filepath=config.ohlcv_history_filepath.format(crypto_symbol),
                gtrends_history_filepath=config.gtrends_history_filepath.format(crypto_symbol),
                dataset_save_filepath=config.dataset_save_filepath.format(crypto_symbol),
                ta_config=ta_config,
                impute_missing_gtrends=impute_missing_gtrends,
                gtrends_imputing_percentage_threshold=gtrends_imputing_percentage_threshold
            )

            if verbose:
                print(f'Successfully has built {crypto_symbol} dataset')
        else:
            if verbose:
                print(f'Download of {crypto_symbol} has been aborted')


if __name__ == '__main__':
    main()
