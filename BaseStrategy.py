from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd


class BaseStrategy(ABC):
    
    def __init__(self):
        self.Parameters = {}
        
    @abstractmethod
    def GetParameters(self) -> Dict[str, Dict[str, Any]]:
        pass
    
    @abstractmethod
    def SetParameters(self, Parameters: Dict[str, Any]) -> None:
        pass
    
    @abstractmethod
    def GenerateSignals(self, Data: pd.DataFrame) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def CalculateFitness(self, BacktestResults: Dict[str, Any]) -> float:
        pass
    
    @abstractmethod
    def GetFitnessMetricName(self) -> str:
        pass