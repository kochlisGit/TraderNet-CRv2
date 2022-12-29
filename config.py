from database.entities.crypto import Crypto
from database.network.coinapi.ohlcv import OHLCVDownloader

# --- Database ---
supported_cryptos = {
    'BTC': Crypto(symbol='BTC', name='bitcoin', start_year=2017),
    'ETH': Crypto(symbol='ETH', name='ethereum', start_year=2017),
    'SOL': Crypto(symbol='SOL',  name='solana', start_year=2020),
    'ADA': Crypto(symbol='ADA', name='ada', start_year=2017),
    'BNB': Crypto(symbol='BNB', name='bnb', start_year=2019),
    'XRP': Crypto(symbol='XRP', name='xrp', start_year=2019),
    'DOGE': Crypto(symbol='DOGE', name='doge', start_year=2020),
    'MATIC': Crypto(symbol='MATIC', name='polygon', start_year=2020),
    'TRON': Crypto(symbol='TRON', name='tron', start_year=2018),
    'LTC': Crypto(symbol='LTC', name='litecoin', start_year=2018),
    'DOT': Crypto(symbol='DOT', name='polkadot', start_year=2021),
    'AVAX': Crypto(symbol='AVAX', name='avalanche', start_year=2021),
    'XMR': Crypto(symbol='XMR', name='monero', start_year=2018),
    'BAT': Crypto(symbol='BAT', name='basic authentication token', start_year=2018),
    'LRC': Crypto(symbol='LRC', name='loopring', start_year=2018)
}

ohlcv_dataset_period_id = OHLCVDownloader.HistoricalFrequency.HOUR
ohlcv_history_filepath = 'database/storage/downloads/ohlcv/{}.csv'
gtrends_history_filepath = 'database/storage/downloads/gtrends/{}.csv'
dataset_save_filepath = 'database/storage/datasets/{}.csv'
all_features = [
    'date', 'open', 'high', 'low', 'close', 'volume', 'trades',
    'open_log_returns', 'high_log_returns', 'low_log_returns',
    'close_log_returns', 'volume_log_returns', 'trades_log_returns', 'hour',
    'dema', 'vwap', 'bband_up', 'bband_down', 'adl', 'obv',
    'macd_signal_diffs', 'stoch', 'aroon_up', 'aroon_down', 'rsi', 'adx', 'cci',
    'close_dema', 'close_vwap', 'bband_up_close', 'close_bband_down', 'adl_diffs2', 'obv_diffs2', 'trends'
]
regression_features = [
    'open_log_returns', 'high_log_returns', 'low_log_returns',
    'close_log_returns', 'volume_log_returns', 'trades_log_returns', 'hour',
    'macd_signal_diffs', 'stoch', 'aroon_up', 'aroon_down', 'rsi', 'adx', 'cci',
    'close_dema', 'close_vwap', 'bband_up_close', 'close_bband_down', 'adl_diffs2', 'obv_diffs2', 'trends'
]

# --- Model ---
checkpoint_dir = 'database/storage/checkpoints/'

# --- Clustering ---
crypto_clusters = [
    ['BTC', 'ETH', 'SOL', 'ADA', 'XPR', 'DOGE', 'DOT', 'AVAX', 'BAT', 'LRC'],
    ['ETH', 'BNB', 'MATIC', 'TRON', 'LTC', 'XMR']
]
