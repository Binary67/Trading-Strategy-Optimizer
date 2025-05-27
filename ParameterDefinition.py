from typing import Any, Union
from enum import Enum


class ParameterType(Enum):
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"


class ParameterDefinition:
    
    def __init__(self, 
                 Name: str,
                 DataType: ParameterType,
                 MinValue: Union[int, float] = None,
                 MaxValue: Union[int, float] = None,
                 StepSize: Union[int, float] = None,
                 DefaultValue: Any = None):
        
        self.Name = Name
        self.DataType = DataType
        self.MinValue = MinValue
        self.MaxValue = MaxValue
        self.StepSize = StepSize
        self.DefaultValue = DefaultValue
        
        self._ValidateParameter()
    
    def _ValidateParameter(self) -> None:
        if self.DataType == ParameterType.BOOL:
            if self.MinValue is not None or self.MaxValue is not None:
                raise ValueError("Boolean parameters should not have min/max values")
            if self.StepSize is not None:
                raise ValueError("Boolean parameters should not have step size")
        else:
            if self.MinValue is None or self.MaxValue is None:
                raise ValueError(f"Numeric parameters must have min and max values")
            if self.MinValue >= self.MaxValue:
                raise ValueError(f"MinValue must be less than MaxValue")
            if self.DataType == ParameterType.INT:
                if not isinstance(self.MinValue, int) or not isinstance(self.MaxValue, int):
                    raise ValueError("Integer parameters must have integer bounds")
                if self.StepSize is not None and not isinstance(self.StepSize, int):
                    raise ValueError("Integer parameters must have integer step size")
    
    def ToDictionary(self) -> dict:
        return {
            "Name": self.Name,
            "DataType": self.DataType.value,
            "MinValue": self.MinValue,
            "MaxValue": self.MaxValue,
            "StepSize": self.StepSize,
            "DefaultValue": self.DefaultValue
        }