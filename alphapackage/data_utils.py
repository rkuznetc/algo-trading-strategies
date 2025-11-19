import os
import pandas as pd

class DataHolder:
    """
        Class, which supports loading and storing OHLCV data from .csv
        Keeps OHLCV data in dictionary

        Args:
            - path (str): path to data directory
            - filter_outliers (boolean): flag, if it outliers in close.csv should be nullyfied

        Methods:
            - get_ohlcv(): get dictionary with OHLCVR data (OHCLV + returns)
            - preprocess(custom_func): performing preprocessing using a custom function to an every dataframe
    """
    def __init__(self, path: str = '../csv_data', filter_outliers: bool = False):
        self.path = path
        self.data = {}
        
        expected_files = ['open.csv', 'high.csv', 'low.csv', 'close.csv', 'volume.csv']
        
        # Load close.csv firstly
        close_path = os.path.join(self.path, 'close.csv')
        if not os.path.exists(close_path):
            raise FileNotFoundError(f"Файл close.csv не найден по пути: {close_path}")
        self.data['close'] = pd.read_csv(close_path, index_col=0, parse_dates=True)
        
        # Optional filtering outlier close
        high_price_tickers = None
        if filter_outliers:
            high_price_tickers = self.filter_outliers()
            if high_price_tickers:
                self.data['close'].drop(columns=high_price_tickers, inplace=True)
        
        # Load every other .csv file (except dropped tickers)
        for filename in os.listdir(self.path):
            if filename.endswith('.csv') and filename != 'close.csv':
                key = filename[:-4]  # transform filename, e.g. "high.csv" -> "high"
                if key not in ['open', 'high', 'low', 'volume']:
                    continue  # ignore non-OHLCV data

                file_path = os.path.join(self.path, filename)
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                if high_price_tickers:
                    df.drop(columns=high_price_tickers, inplace=True)
                self.data[key] = df
        
        # Check, if all OHLCV data was loaded
        missing = [f for f in expected_files if f[:-4] not in self.data]
        if missing:
            raise ValueError(f"Missing files: {missing}")

        # Calculate returns dataframe
        self.data['return'] = self.data['close'].pct_change()
    

    def filter_outliers(self) -> list[str]:
        high_price_mask = self.data['close'].iloc[0] > 1e5
        return high_price_mask[high_price_mask].index.tolist()
    

    def get_ohlcv(self) -> dict:
        return self.data.copy()
    

    def preprocess(self, custom_func=None):
        for key, df in self.data.items():
            if custom_func:
                custom_func(df)