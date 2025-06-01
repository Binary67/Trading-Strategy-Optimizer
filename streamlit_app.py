import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import traceback # For error logging

# Import necessary project modules
from DataDownloader import YFinanceDownloader
from GeneticAlgorithm import GeneticAlgorithm
from ExampleStrategies import MovingAverageCrossoverStrategy, RSIStrategy, BollingerBandsStrategy
from BaseStrategy import BaseStrategy
from ParameterDefinition import ParameterType
from Backtesting import SimpleBacktest

# --- Initialize Session State ---
# Ensure all session state variables are initialized to avoid errors on first run or after resets.
if 'OptimizationRun' not in st.session_state:
    st.session_state.OptimizationRun = False
if 'BestParameters' not in st.session_state:
    st.session_state.BestParameters = None
if 'BestFitness' not in st.session_state:
    st.session_state.BestFitness = None
if 'FinalBacktestResults' not in st.session_state:
    st.session_state.FinalBacktestResults = None
if 'GenerationHistory' not in st.session_state:
    st.session_state.GenerationHistory = []
if 'TickerSymbolForDisplay' not in st.session_state:
    st.session_state.TickerSymbolForDisplay = ""
if 'SelectedStrategyNameForDisplay' not in st.session_state:
    st.session_state.SelectedStrategyNameForDisplay = ""
if 'FitnessMetricName' not in st.session_state:
    st.session_state.FitnessMetricName = "Fitness"


# --- App Title and Introduction ---
st.title("Trading Strategy Genetic Algorithm Optimizer")

st.markdown("""
### Welcome to the Trading Strategy Genetic Algorithm Optimizer!
This tool allows you to optimize trading strategy parameters using a Genetic Algorithm.

**How to use:**
1.  **Configure Settings (Sidebar):**
    *   **Data:** Enter a Ticker Symbol (e.g., AAPL, MSFT), select a Start Date, End Date, and Data Interval for historical market data.
    *   **Strategy:** Choose a predefined trading strategy (e.g., Moving Average Crossover). The default parameters for the selected strategy will be shown, and these define the search space for the GA.
    *   **Genetic Algorithm:** Adjust parameters like Population Size, Number of Generations, Mutation Rate, Crossover Rate, and Elite Size (number of top individuals to carry over to the next generation).
2.  **Run Optimization:** Click the "Run Optimization" button. The process may take some time depending on the data size and GA settings.
3.  **View Results:** Once complete, the optimized strategy parameters, key performance metrics (Total Return, Sharpe Ratio, Max Drawdown), an equity curve chart, and a plot of the GA's fitness improvement over generations will be displayed.
""")
st.divider()


# --- Sidebar for User Inputs ---
st.sidebar.header("Data and Strategy Settings")
TickerSymbol = st.sidebar.text_input("Ticker Symbol", value="AAPL", help="Enter a valid stock ticker symbol (e.g., AAPL, GOOGL).")

DefaultEndDate = datetime.today()
DefaultStartDate = DefaultEndDate - timedelta(days=365)
StartDateInput = st.sidebar.date_input("Start Date", value=DefaultStartDate, help="Start date for historical data.")
EndDateInput = st.sidebar.date_input("End Date", value=DefaultEndDate, help="End date for historical data. Must be after Start Date.")

IntervalOptions = ['1d', '1wk', '1mo', '1h', '30m', '15m', '5m', '2m', '1m']
Interval = st.sidebar.selectbox("Interval", options=IntervalOptions, index=IntervalOptions.index('1h'), help="Frequency of historical data.")

StrategyClasses = {
    "Moving Average Crossover": MovingAverageCrossoverStrategy,
    "RSI Strategy": RSIStrategy,
    "Bollinger Bands Strategy": BollingerBandsStrategy
}
SelectedStrategyName = st.sidebar.selectbox("Choose Strategy", options=list(StrategyClasses.keys()), help="Select the trading strategy to optimize.")
SelectedStrategyClass = StrategyClasses[SelectedStrategyName]
CurrentStrategyInstance = SelectedStrategyClass() # Used for displaying parameters and getting fitness metric name

st.sidebar.header(f"{SelectedStrategyName} Parameters")
StrategyParameters = CurrentStrategyInstance.GetParameters()
UserInputStrategyParameters = {} # Note: These UI parameters are for display; GA uses ranges from strategy definition.

for ParamName, ParamInfo in StrategyParameters.items():
    DefaultValue = ParamInfo['DefaultValue']
    MinValue = ParamInfo['MinValue']
    MaxValue = ParamInfo['MaxValue']
    StepSize = ParamInfo['StepSize']
    DataType = ParamInfo['DataType']
    HelpText = ParamInfo.get('Description', f"Adjust {ParamName}") # Assuming ParameterDefinition might have 'Description'

    if DataType == ParameterType.INT.value:
        UserInputStrategyParameters[ParamName] = st.sidebar.number_input(
            label=ParamName, min_value=MinValue, max_value=MaxValue, value=DefaultValue, step=StepSize,
            key=f"strat_{SelectedStrategyName}_{ParamName}", help=HelpText
        )
    elif DataType == ParameterType.FLOAT.value:
        UserInputStrategyParameters[ParamName] = st.sidebar.number_input(
            label=ParamName, min_value=float(MinValue), max_value=float(MaxValue), value=float(DefaultValue), step=float(StepSize),
            format="%.2f", key=f"strat_{SelectedStrategyName}_{ParamName}", help=HelpText
        )
    elif DataType == ParameterType.BOOL.value:
        UserInputStrategyParameters[ParamName] = st.sidebar.checkbox(
            label=ParamName, value=DefaultValue, key=f"strat_{SelectedStrategyName}_{ParamName}", help=HelpText
        )

st.sidebar.header("Genetic Algorithm Settings")
PopulationSize = st.sidebar.number_input("Population Size", min_value=10, max_value=1000, value=50, step=10, help="Number of individuals (parameter sets) in each generation.")
GenerationCount = st.sidebar.number_input("Generation Count", min_value=5, max_value=500, value=20, step=5, help="Number of generations the GA will run.")
MutationRate = st.sidebar.slider("Mutation Rate", min_value=0.01, max_value=1.0, value=0.1, step=0.01, help="Probability of a parameter mutating.")
CrossoverRate = st.sidebar.slider("Crossover Rate", min_value=0.1, max_value=1.0, value=0.8, step=0.01, help="Probability of two individuals combining parameters.")
# EliteSize max_value calculation is robust: max(1, PopulationSize // 5 if PopulationSize > 0 else 1)
MaxEliteSize = max(1, PopulationSize // 5 if PopulationSize > 0 else 1)
EliteSize = st.sidebar.number_input("Elite Size", min_value=1, max_value=MaxEliteSize, value=min(5, MaxEliteSize), step=1, help="Number of best individuals to carry to the next generation unchanged.")

# --- Run Optimization Button ---
RunOptimizationButton = st.button("Run Optimization", help="Click to start the optimization process.")

# --- Main Area for Status Messages and Results ---
StatusPlaceholder = st.empty() # Used for status messages during optimization

if RunOptimizationButton:
    # --- 0. Reset Session State for New Run ---
    st.session_state.OptimizationRun = False
    st.session_state.BestParameters = None
    st.session_state.BestFitness = None
    st.session_state.FinalBacktestResults = None
    st.session_state.GenerationHistory = []
    st.session_state.TickerSymbolForDisplay = ""
    st.session_state.SelectedStrategyNameForDisplay = ""
    st.session_state.FitnessMetricName = "Fitness" # Reset to default
    StatusPlaceholder.empty() # Clear previous status messages
    TradingDataTable = pd.DataFrame() # Initialize to ensure it's defined in this scope

    # --- 1. Validate Inputs ---
    StatusPlaceholder.info("Validating inputs...")
    if StartDateInput >= EndDateInput:
        StatusPlaceholder.error("❌ Error: Start Date must be before End Date.")
        st.stop() # Halt execution for this run

    # --- 2. Download Market Data ---
    DataDownloadMessage = f"Downloading data for {TickerSymbol} ({StartDateInput.strftime('%Y-%m-%d')} to {EndDateInput.strftime('%Y-%m-%d')} at {Interval} interval)..."
    StatusPlaceholder.info(f"⏳ {DataDownloadMessage}")
    try:
        with st.spinner(DataDownloadMessage):
            DataDownloader = YFinanceDownloader(TickerSymbol, StartDateInput.strftime('%Y-%m-%d'), EndDateInput.strftime('%Y-%m-%d'), Interval)
            TradingDataTable = DataDownloader.DownloadData()

        if TradingDataTable.empty:
            StatusPlaceholder.error(f"❌ Failed to download data for {TickerSymbol} or data is empty. Please check ticker, date range, and interval. Ensure the ticker is valid and data exists for the selected period.")
            st.stop()

        StatusPlaceholder.success(f"✅ Data for {TickerSymbol} downloaded successfully. Shape: {TradingDataTable.shape}")

    except Exception as e:
        StatusPlaceholder.error(f"❌ Error downloading data for {TickerSymbol}: {e}")
        st.error(traceback.format_exc()) # Provide full traceback for debugging
        st.stop()

    # --- 3. Setup Genetic Algorithm and Strategy ---
    StatusPlaceholder.info("⏳ Setting up Genetic Algorithm and Strategy...")
    try:
        TempStrategyInstance = SelectedStrategyClass() # Instance to get fitness metric name
        FitnessMetricName = TempStrategyInstance.GetFitnessMetricName()
        st.session_state.FitnessMetricName = FitnessMetricName # Store for display

        GA = GeneticAlgorithm(
            PopulationSize=PopulationSize,
            GenerationCount=GenerationCount,
            MutationRate=MutationRate,
            CrossoverRate=CrossoverRate,
            EliteSize=EliteSize
        )
        StatusPlaceholder.info(f"✅ Genetic Algorithm configured. Optimizing for: {FitnessMetricName}.")
    except Exception as e:
        StatusPlaceholder.error(f"❌ Error setting up Genetic Algorithm: {e}")
        st.error(traceback.format_exc())
        st.stop()

    # --- 4. Run Optimization ---
    OptimizationMessage = f"Optimizing {SelectedStrategyName} for {TickerSymbol} (Fitness: {FitnessMetricName})... Generations: {GenerationCount}, Population: {PopulationSize}. This may take a moment."
    StatusPlaceholder.info(f"⏳ {OptimizationMessage}")
    try:
        with st.spinner(OptimizationMessage):
            OptimizationResults = GA.Optimize(
                StrategyClass=SelectedStrategyClass,
                TradingData=TradingDataTable,
                BacktestFunction=SimpleBacktest
            )

        StatusPlaceholder.success("✅ Optimization complete!")

        # --- 5. Process and Store Results ---
        StatusPlaceholder.info("⏳ Processing and storing results...")
        BestParameters = OptimizationResults['BestParameters']
        BestFitness = OptimizationResults['BestFitness']
        GenerationHistoryList = OptimizationResults.get('History', [])

        OptimizedStrategyInstance = SelectedStrategyClass()
        OptimizedStrategyInstance.SetParameters(BestParameters)
        FinalBacktestResults = SimpleBacktest(OptimizedStrategyInstance, TradingDataTable)

        st.session_state.OptimizationRun = True
        st.session_state.BestParameters = BestParameters
        st.session_state.BestFitness = BestFitness
        st.session_state.FinalBacktestResults = FinalBacktestResults
        st.session_state.GenerationHistory = GenerationHistoryList
        st.session_state.TickerSymbolForDisplay = TickerSymbol
        st.session_state.SelectedStrategyNameForDisplay = SelectedStrategyName

        StatusPlaceholder.success("✅ Processing complete! Results are updated below.")
        st.rerun() # Rerun to update the results display immediately

    except Exception as e:
        StatusPlaceholder.error(f"❌ An error occurred during optimization: {e}")
        st.error(traceback.format_exc())
        st.session_state.OptimizationRun = False # Ensure flag is False on error


# --- Optimization Results Display Area ---
st.header("Optimization Results")

if st.session_state.get('OptimizationRun', False):
    BestParameters = st.session_state.BestParameters
    BestFitness = st.session_state.BestFitness
    FinalBacktestResults = st.session_state.FinalBacktestResults
    GenerationHistoryList = st.session_state.GenerationHistory
    TickerSymbol = st.session_state.TickerSymbolForDisplay
    SelectedStrategyName = st.session_state.SelectedStrategyNameForDisplay
    FitnessMetricName = st.session_state.FitnessMetricName

    if FinalBacktestResults is None:
        st.warning("Optimization run, but final results are not available. Please try running again.")
    else:
        st.subheader(f"Results for {TickerSymbol} using {SelectedStrategyName}")

        st.subheader("Optimized Strategy Parameters")
        if BestParameters:
            st.json(BestParameters, expanded=True) # Show expanded by default
        else:
            st.write("No parameters found.")

        st.subheader("Optimized Strategy Performance")
        Col1, Col2, Col3 = st.columns(3)
        Col1.metric("Total Return", f"{FinalBacktestResults.get('TotalReturn', 0):.2%}", help="Total percentage return over the backtest period.")
        Col2.metric(f"Backtest {FitnessMetricName}", f"{FinalBacktestResults.get('SharpeRatio', 0):.2f}", help=f"{FitnessMetricName} calculated on the final backtest with optimized parameters. Higher is generally better.")
        Col3.metric("Max Drawdown", f"{FinalBacktestResults.get('MaxDrawdown', 0):.2%}", help="Largest peak-to-trough decline (loss) during the backtest period. Lower is generally better.")

        st.metric(f"Best Genetic Algorithm Fitness ({FitnessMetricName})", f"{BestFitness:.4f}", help=f"The highest fitness score ({FitnessMetricName}) achieved by any individual strategy during the genetic algorithm optimization process. This score guided the selection of the 'Optimized Strategy Parameters'.")

        st.subheader("Equity Curve (Cumulative Returns)")
        CumulativeReturnsData = FinalBacktestResults.get('CumulativeReturns')
        if CumulativeReturnsData is not None and not CumulativeReturnsData.empty:
            st.line_chart(CumulativeReturnsData)
        else:
            st.write("Cumulative returns data not available for plotting.")

        st.subheader("Genetic Algorithm Progress")
        if GenerationHistoryList:
            HistoryDf = pd.DataFrame(GenerationHistoryList)
            if not HistoryDf.empty and all(col in HistoryDf.columns for col in ['Generation', 'BestFitness', 'AverageFitness']):
                st.line_chart(HistoryDf, x='Generation', y=['BestFitness', 'AverageFitness'], color=['#00FF00', '#0000FF']) # Best in Green, Avg in Blue
                with st.expander("View Generation History Data Table"):
                    st.dataframe(HistoryDf)
            elif not HistoryDf.empty:
                 st.warning("Generation history data available but may be missing expected columns (Generation, BestFitness, AverageFitness). Displaying raw data.")
                 st.dataframe(HistoryDf)
            else:
                st.info("Generation history data is empty.")
        else:
            st.info("No generation history was recorded for this optimization run.")
else:
    st.info("ℹ️ Run an optimization to view results. Configure settings in the sidebar and click 'Run Optimization'.")

```
