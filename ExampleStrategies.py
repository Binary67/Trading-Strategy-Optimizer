import pandas as pd
from typing import Dict, Any
from BaseStrategy import BaseStrategy
from ParameterDefinition import ParameterDefinition, ParameterType


class MovingAverageCrossoverStrategy(BaseStrategy):
    
    def __init__(self):
        super().__init__()
        self.FastPeriod = 10
        self.SlowPeriod = 50
        
    def GetParameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "FastPeriod": ParameterDefinition(
                Name="FastPeriod",
                DataType=ParameterType.INT,
                MinValue=5,
                MaxValue=50,
                StepSize=1,
                DefaultValue=10
            ).ToDictionary(),
            "SlowPeriod": ParameterDefinition(
                Name="SlowPeriod",
                DataType=ParameterType.INT,
                MinValue=20,
                MaxValue=200,
                StepSize=1,
                DefaultValue=50
            ).ToDictionary()
        }
    
    def SetParameters(self, Parameters: Dict[str, Any]) -> None:
        self.FastPeriod = Parameters.get("FastPeriod", self.FastPeriod)
        self.SlowPeriod = Parameters.get("SlowPeriod", self.SlowPeriod)
        
    def GenerateSignals(self, Data: pd.DataFrame) -> pd.DataFrame:
        Signals = Data.copy()
        Signals['FastMA'] = Data['Close'].rolling(window=self.FastPeriod).mean()
        Signals['SlowMA'] = Data['Close'].rolling(window=self.SlowPeriod).mean()
        
        Signals['Signal'] = 0
        Signals.loc[Signals['FastMA'] > Signals['SlowMA'], 'Signal'] = 1
        Signals.loc[Signals['FastMA'] < Signals['SlowMA'], 'Signal'] = -1
        
        return Signals
    
    def CalculateFitness(self, BacktestResults: Dict[str, Any]) -> float:
        SharpeRatio = BacktestResults.get('SharpeRatio', 0.0)
        TotalReturn = BacktestResults.get('TotalReturn', 0.0)
        MaxDrawdown = BacktestResults.get('MaxDrawdown', 0.0)
        
        WeightSharpe = 0.4
        WeightReturn = 0.5
        WeightDrawdown = 0.1
        
        NormalizedReturn = TotalReturn * 100
        DrawdownPenalty = 1 - MaxDrawdown
        
        FitnessScore = (WeightSharpe * SharpeRatio + 
                       WeightReturn * NormalizedReturn + 
                       WeightDrawdown * DrawdownPenalty)
        
        return FitnessScore
    
    def GetFitnessMetricName(self) -> str:
        return "Weighted Score"


class RSIStrategy(BaseStrategy):
    
    def __init__(self):
        super().__init__()
        self.Period = 14
        self.OverboughtLevel = 70
        self.OversoldLevel = 30
        
    def GetParameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "Period": ParameterDefinition(
                Name="Period",
                DataType=ParameterType.INT,
                MinValue=5,
                MaxValue=30,
                StepSize=1,
                DefaultValue=14
            ).ToDictionary(),
            "OverboughtLevel": ParameterDefinition(
                Name="OverboughtLevel",
                DataType=ParameterType.FLOAT,
                MinValue=60.0,
                MaxValue=90.0,
                StepSize=1.0,
                DefaultValue=70.0
            ).ToDictionary(),
            "OversoldLevel": ParameterDefinition(
                Name="OversoldLevel",
                DataType=ParameterType.FLOAT,
                MinValue=10.0,
                MaxValue=40.0,
                StepSize=1.0,
                DefaultValue=30.0
            ).ToDictionary()
        }
    
    def SetParameters(self, Parameters: Dict[str, Any]) -> None:
        self.Period = Parameters.get("Period", self.Period)
        self.OverboughtLevel = Parameters.get("OverboughtLevel", self.OverboughtLevel)
        self.OversoldLevel = Parameters.get("OversoldLevel", self.OversoldLevel)
        
    def GenerateSignals(self, Data: pd.DataFrame) -> pd.DataFrame:
        Signals = Data.copy()
        
        Delta = Data['Close'].diff()
        Gain = (Delta.where(Delta > 0, 0)).rolling(window=self.Period).mean()
        Loss = (-Delta.where(Delta < 0, 0)).rolling(window=self.Period).mean()
        
        RS = Gain / Loss
        Signals['RSI'] = 100 - (100 / (1 + RS))
        
        Signals['Signal'] = 0
        Signals.loc[Signals['RSI'] < self.OversoldLevel, 'Signal'] = 1
        Signals.loc[Signals['RSI'] > self.OverboughtLevel, 'Signal'] = -1
        
        return Signals
    
    def CalculateFitness(self, BacktestResults: Dict[str, Any]) -> float:
        return BacktestResults.get('TotalReturn', 0.0)
    
    def GetFitnessMetricName(self) -> str:
        return "Total Return"


class BollingerBandsStrategy(BaseStrategy):
    
    def __init__(self):
        super().__init__()
        self.Period = 20
        self.StdDevMultiplier = 2.0
        
    def GetParameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "Period": ParameterDefinition(
                Name="Period",
                DataType=ParameterType.INT,
                MinValue=10,
                MaxValue=50,
                StepSize=1,
                DefaultValue=20
            ).ToDictionary(),
            "StdDevMultiplier": ParameterDefinition(
                Name="StdDevMultiplier",
                DataType=ParameterType.FLOAT,
                MinValue=1.0,
                MaxValue=3.0,
                StepSize=0.1,
                DefaultValue=2.0
            ).ToDictionary()
        }
    
    def SetParameters(self, Parameters: Dict[str, Any]) -> None:
        self.Period = Parameters.get("Period", self.Period)
        self.StdDevMultiplier = Parameters.get("StdDevMultiplier", self.StdDevMultiplier)
        
    def GenerateSignals(self, Data: pd.DataFrame) -> pd.DataFrame:
        Signals = Data.copy()
        
        Signals['MA'] = Data['Close'].rolling(window=self.Period).mean()
        Signals['StdDev'] = Data['Close'].rolling(window=self.Period).std()
        
        Signals['UpperBand'] = Signals['MA'] + (Signals['StdDev'] * self.StdDevMultiplier)
        Signals['LowerBand'] = Signals['MA'] - (Signals['StdDev'] * self.StdDevMultiplier)
        
        Signals['Signal'] = 0
        Signals.loc[Data['Close'] < Signals['LowerBand'], 'Signal'] = 1
        Signals.loc[Data['Close'] > Signals['UpperBand'], 'Signal'] = -1
        
        return Signals
    
    def CalculateFitness(self, BacktestResults: Dict[str, Any]) -> float:
        SharpeRatio = BacktestResults.get('SharpeRatio', 0.0)
        MaxDrawdown = BacktestResults.get('MaxDrawdown', -1.0)
        return SharpeRatio - abs(MaxDrawdown)
    
    def GetFitnessMetricName(self) -> str:
        return "Sharpe Ratio - Max Drawdown"