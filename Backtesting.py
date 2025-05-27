import numpy as np


def SimpleBacktest(Strategy, TradingData):
    Signals = Strategy.GenerateSignals(TradingData)
    
    Returns = TradingData['Close'].pct_change().fillna(0)
    
    StrategyReturns = Returns * Signals['Signal'].shift(1).fillna(0)
    
    CumulativeReturns = (1 + StrategyReturns).cumprod()
    TotalReturn = CumulativeReturns.iloc[-1] - 1
    
    DailyReturns = StrategyReturns.resample('D').sum()
    SharpeRatio = np.sqrt(252) * DailyReturns.mean() / (DailyReturns.std() + 1e-6)
    
    RunningMax = CumulativeReturns.expanding().max()
    Drawdown = (CumulativeReturns - RunningMax) / RunningMax
    MaxDrawdown = Drawdown.min()
    
    return {
        'SharpeRatio': SharpeRatio,
        'TotalReturn': TotalReturn,
        'MaxDrawdown': MaxDrawdown
    }