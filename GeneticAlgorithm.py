import numpy as np
import random
from typing import Dict, List, Any, Callable
import pandas as pd


class GeneticAlgorithm:
    def __init__(self, PopulationSize: int = 50, GenerationCount: int = 100,
                 MutationRate: float = 0.1, CrossoverRate: float = 0.8,
                 EliteSize: int = 5):
        self.PopulationSize = PopulationSize
        self.GenerationCount = GenerationCount
        self.MutationRate = MutationRate
        self.CrossoverRate = CrossoverRate
        self.EliteSize = EliteSize
        self.Population = []
        self.BestIndividual = None
        self.BestFitness = -float('inf')
        self.GenerationHistory = []
        
    def InitializePopulation(self, ParameterDefinitions: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        self.ParameterDefinitions = ParameterDefinitions
        self.Population = []
        
        for _ in range(self.PopulationSize):
            Individual = {}
            for ParamName, ParamInfo in ParameterDefinitions.items():
                if ParamInfo['DataType'] == 'int':
                    Value = random.randint(ParamInfo['MinValue'], ParamInfo['MaxValue'])
                elif ParamInfo['DataType'] == 'float':
                    Value = random.uniform(ParamInfo['MinValue'], ParamInfo['MaxValue'])
                elif ParamInfo['DataType'] == 'bool':
                    Value = random.choice([True, False])
                Individual[ParamName] = Value
            self.Population.append(Individual)
            
        return self.Population
    
    def EvaluatePopulation(self, FitnessFunction: Callable[[Dict[str, Any]], float]) -> List[float]:
        FitnessScores = []
        for Individual in self.Population:
            Fitness = FitnessFunction(Individual)
            FitnessScores.append(Fitness)
            
            if Fitness > self.BestFitness:
                self.BestFitness = Fitness
                self.BestIndividual = Individual.copy()
                
        return FitnessScores
    
    def SelectParents(self, FitnessScores: List[float]) -> List[Dict[str, Any]]:
        SortedIndices = np.argsort(FitnessScores)[::-1]
        
        Elite = [self.Population[idx].copy() for idx in SortedIndices[:self.EliteSize]]
        
        TournamentSize = 3
        Parents = Elite.copy()
        
        while len(Parents) < self.PopulationSize:
            TournamentIndices = random.sample(range(len(self.Population)), TournamentSize)
            TournamentFitness = [FitnessScores[idx] for idx in TournamentIndices]
            WinnerIdx = TournamentIndices[np.argmax(TournamentFitness)]
            Parents.append(self.Population[WinnerIdx].copy())
            
        return Parents
    
    def Crossover(self, Parent1: Dict[str, Any], Parent2: Dict[str, Any]) -> tuple:
        if random.random() > self.CrossoverRate:
            return Parent1.copy(), Parent2.copy()
        
        Child1, Child2 = {}, {}
        
        for ParamName in Parent1.keys():
            if random.random() < 0.5:
                Child1[ParamName] = Parent1[ParamName]
                Child2[ParamName] = Parent2[ParamName]
            else:
                Child1[ParamName] = Parent2[ParamName]
                Child2[ParamName] = Parent1[ParamName]
                
        return Child1, Child2
    
    def Mutate(self, Individual: Dict[str, Any]) -> Dict[str, Any]:
        MutatedIndividual = Individual.copy()
        
        for ParamName, ParamInfo in self.ParameterDefinitions.items():
            if random.random() < self.MutationRate:
                if ParamInfo['DataType'] == 'int':
                    MutationRange = int((ParamInfo['MaxValue'] - ParamInfo['MinValue']) * 0.1)
                    MutationAmount = random.randint(-MutationRange, MutationRange)
                    NewValue = MutatedIndividual[ParamName] + MutationAmount
                    NewValue = max(ParamInfo['MinValue'], min(ParamInfo['MaxValue'], NewValue))
                    MutatedIndividual[ParamName] = int(NewValue)
                    
                elif ParamInfo['DataType'] == 'float':
                    MutationRange = (ParamInfo['MaxValue'] - ParamInfo['MinValue']) * 0.1
                    MutationAmount = random.uniform(-MutationRange, MutationRange)
                    NewValue = MutatedIndividual[ParamName] + MutationAmount
                    NewValue = max(ParamInfo['MinValue'], min(ParamInfo['MaxValue'], NewValue))
                    MutatedIndividual[ParamName] = NewValue
                    
                elif ParamInfo['DataType'] == 'bool':
                    MutatedIndividual[ParamName] = not MutatedIndividual[ParamName]
                    
        return MutatedIndividual
    
    def CreateNextGeneration(self, Parents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        NextGeneration = []
        
        NextGeneration.extend(Parents[:self.EliteSize])
        
        while len(NextGeneration) < self.PopulationSize:
            Parent1Idx = random.randint(0, len(Parents) - 1)
            Parent2Idx = random.randint(0, len(Parents) - 1)
            
            Parent1 = Parents[Parent1Idx]
            Parent2 = Parents[Parent2Idx]
            
            Child1, Child2 = self.Crossover(Parent1, Parent2)
            
            Child1 = self.Mutate(Child1)
            Child2 = self.Mutate(Child2)
            
            NextGeneration.append(Child1)
            if len(NextGeneration) < self.PopulationSize:
                NextGeneration.append(Child2)
                
        return NextGeneration[:self.PopulationSize]
    
    def Optimize(self, StrategyClass: Any, TradingData: pd.DataFrame, 
                 BacktestFunction: Callable[[Any, pd.DataFrame], Dict[str, float]]) -> Dict[str, Any]:
        Strategy = StrategyClass()
        ParameterDefinitions = Strategy.GetParameters()
        
        self.InitializePopulation(ParameterDefinitions)
        
        # Get fitness metric name from strategy
        FitnessMetric = Strategy.GetFitnessMetricName()
        
        def FitnessFunction(Parameters: Dict[str, Any]) -> float:
            Strategy.SetParameters(Parameters)
            BacktestResults = BacktestFunction(Strategy, TradingData)
            return Strategy.CalculateFitness(BacktestResults)
        
        for Generation in range(self.GenerationCount):
            FitnessScores = self.EvaluatePopulation(FitnessFunction)
            
            GenerationStats = {
                'Generation': Generation,
                'BestFitness': max(FitnessScores),
                'AverageFitness': np.mean(FitnessScores),
                'WorstFitness': min(FitnessScores)
            }
            self.GenerationHistory.append(GenerationStats)
            
            if Generation % 10 == 0:
                print(f"Generation {Generation}: Best {FitnessMetric} = {GenerationStats['BestFitness']:.4f}, "
                      f"Avg {FitnessMetric} = {GenerationStats['AverageFitness']:.4f}")
            
            Parents = self.SelectParents(FitnessScores)
            self.Population = self.CreateNextGeneration(Parents)
            
        print(f"\nOptimization Complete!")
        print(f"Best {FitnessMetric}: {self.BestFitness:.4f}")
        print(f"Best Parameters: {self.BestIndividual}")
        
        return {
            'BestParameters': self.BestIndividual,
            'BestFitness': self.BestFitness,
            'History': self.GenerationHistory
        }