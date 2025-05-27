from DataDownloader import YFinanceDownloader
from ExampleStrategies import MovingAverageCrossoverStrategy
from StrategyOptimizer import OptimizeStrategy
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    
    print("Downloading market data...")
    DataDownloaderObj = YFinanceDownloader('AAPL', '2023-06-01', '2025-05-15', '1h')
    TradingData = DataDownloaderObj.DownloadData()
    print(f"Data shape: {TradingData.shape}")
    
    Strategies = [
        MovingAverageCrossoverStrategy,
    ]
    
    print("\n" + "="*50)
    print("GENETIC ALGORITHM OPTIMIZATION")
    print("="*50)
    
    for StrategyClass in Strategies:
        OptimizeStrategy(StrategyClass, TradingData)