from GeneticAlgorithm import GeneticAlgorithm
from Backtesting import SimpleBacktest
import json
from datetime import datetime


def OptimizeStrategy(StrategyClass, TradingData):
    print(f"\n{'='*100}")
    print(f"Optimizing {StrategyClass.__name__} with Genetic Algorithm")
    print('='*100)
    
    GA = GeneticAlgorithm(
        PopulationSize = 500,
        GenerationCount = 150,
        MutationRate = 0.1,
        CrossoverRate = 0.8,
        EliteSize=3
    )
    
    OptimizationResults = GA.Optimize(
        StrategyClass=StrategyClass,
        TradingData=TradingData,
        BacktestFunction=SimpleBacktest
    )
    
    print("\n" + "="*40)
    print("Optimization Results:")
    print("="*40)
    print(f"Best Parameters: {OptimizationResults['BestParameters']}")
    print(f"Best Fitness: {OptimizationResults['BestFitness']:.4f}")
    
    OptimizedStrategy = StrategyClass()
    OptimizedStrategy.SetParameters(OptimizationResults['BestParameters'])
    OptimizedBacktest = SimpleBacktest(OptimizedStrategy, TradingData)
    print(f"\nOptimized Performance:")
    print(f"  Sharpe Ratio: {OptimizedBacktest['SharpeRatio']:.4f}")
    print(f"  Total Return: {OptimizedBacktest['TotalReturn']:.4%}")
    print(f"  Max Drawdown: {OptimizedBacktest['MaxDrawdown']:.4%}")
    
    DefaultStrategy = StrategyClass()
    DefaultBacktest = SimpleBacktest(DefaultStrategy, TradingData)
    print(f"\nDefault Performance:")
    print(f"  Sharpe Ratio: {DefaultBacktest['SharpeRatio']:.4f}")
    print(f"  Total Return: {DefaultBacktest['TotalReturn']:.4%}")
    print(f"  Max Drawdown: {DefaultBacktest['MaxDrawdown']:.4%}")
    
    # Create results dictionary
    ResultsData = {
        "StrategyName": StrategyClass.__name__,
        "OptimizationTimestamp": datetime.now().isoformat(),
        "BestParameters": OptimizationResults['BestParameters'],
        "BestFitness": OptimizationResults['BestFitness'],
        "OptimizedPerformance": {
            "SharpeRatio": OptimizedBacktest['SharpeRatio'],
            "TotalReturn": OptimizedBacktest['TotalReturn'],
            "MaxDrawdown": OptimizedBacktest['MaxDrawdown']
        },
        "DefaultPerformance": {
            "SharpeRatio": DefaultBacktest['SharpeRatio'],
            "TotalReturn": DefaultBacktest['TotalReturn'],
            "MaxDrawdown": DefaultBacktest['MaxDrawdown']
        }
    }
    
    # Save to JSON file
    FileName = f"PerformanceMetrics_{StrategyClass.__name__}_{datetime.now().strftime('%Y%m%d')}.json"
    with open(FileName, 'w') as JsonFile:
        json.dump(ResultsData, JsonFile, indent=4)
    
    print(f"\n✓ Results saved to: {FileName}")