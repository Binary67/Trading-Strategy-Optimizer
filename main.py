from DataDownloader import YFinanceDownloader
import pandas as pd
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    
    DataDownloaderObj = YFinanceDownloader('AAPL', '2023-06-01', '2025-05-15', '1h')
    TradingData = DataDownloaderObj.DownloadData()