# Monte Carlo Stock Price Simulation - Documentation

## Project Overview

This project implements a Monte Carlo simulation to model potential future stock price movements based on historical price data. The simulation generates multiple possible price paths to quantify uncertainty and provide probabilistic forecasts rather than single-point predictions.

## Author

Created as a learning project to understand Monte Carlo methods and their applications in financial modeling.

---

## Table of Contents

1. [Requirements](#requirements)
2. [How It Works](#how-it-works)
3. [Code Structure](#code-structure)
4. [Key Parameters](#key-parameters)
5. [Understanding the Output](#understanding-the-output)
6. [Usage Instructions](#usage-instructions)
7. [Validation Approach](#validation-approach)
8. [Limitations](#limitations)
9. [Future Improvements](#future-improvements)

---

## Requirements

### Python Libraries

```python
yfinance       # For downloading historical stock data
numpy          # For numerical computations
pandas         # For data manipulation
matplotlib     # For visualization
```

### Installation

```bash
pip install yfinance numpy pandas matplotlib
```

---

## How It Works

### The Monte Carlo Method

Monte Carlo simulation uses repeated random sampling to model uncertainty. For stock prices, the process:

1. **Analyzes historical data** to calculate average daily returns and volatility
2. **Generates random price movements** based on these statistics
3. **Simulates many possible futures** (e.g., 100 different scenarios)
4. **Aggregates results** to understand the range of likely outcomes

### Mathematical Foundation

Each day's price change is modeled as:

```
New Price = Current Price × (1 + random_return)
```

Where:

```
random_return = mean_daily_return + (volatility × random_shock)
random_shock ~ N(0, 1)  # Standard normal distribution
```

This follows the assumption that stock returns are approximately normally distributed (though this is a simplification of reality).

---

## Code Structure

### 1. Data Collection

```python
ticker = 'AAPL'
data = yf.download(ticker, start='2022-06-01', end='2023-01-06')
```

- Downloads historical price data for the specified stock
- Date range determines the historical period used for calculating statistics

### 2. Calculate Historical Statistics

```python
data['Daily Return'] = (data['Close'] - data['Close'].shift(1)) / data['Close'].shift(1)
mean = data['Daily Return'].mean()
std = data['Daily Return'].std()
```

- **Daily Return**: Percentage change in price from one day to the next
- **Mean**: Average daily return (drift)
- **Standard Deviation (std)**: Volatility measure - how much prices typically fluctuate

### 3. Monte Carlo Simulation

```python
for i in range(num_simulations):
    price = starting_price
    for j in range(num_days):
        daily_returns = np.random.normal(mean, std)
        price = price * (1 + daily_returns)
        simulations[i][j] = price
```

- **Outer loop**: Creates multiple independent simulations
- **Inner loop**: Simulates each day's price change
- Each simulation represents one possible future price path

### 4. Analysis

```python
final_prices = simulations[:, -1]
mean_final_price = np.mean(final_prices)
lower_bound = np.percentile(final_prices, 5)
upper_bound = np.percentile(final_prices, 95)
```

- Extracts final prices from all simulation paths
- Calculates 90% confidence interval (5th to 95th percentile)
- This means: "There's a 90% chance the actual price will fall within this range"

### 5. Visualization

Creates a comprehensive plot showing:

- All 100 simulation paths (light blue, transparent)
- Mean path (red line)
- Starting price reference (green dashed line)
- 90% confidence interval boundaries (orange dashed lines)
- Statistics box with key metrics

---

## Key Parameters

### Adjustable Settings

| Parameter         | Current Value | Description                       |
| ----------------- | ------------- | --------------------------------- |
| `ticker`          | 'AAPL'        | Stock symbol to simulate          |
| `start`           | '2022-06-01'  | Historical data start date        |
| `end`             | '2023-01-06'  | Historical data end date          |
| `num_simulations` | 100           | Number of price paths to generate |
| `num_days`        | 5             | Trading days to simulate forward  |

### Modifying Parameters

**To change the stock:**

```python
ticker = 'MSFT'  # or any valid ticker
```

**To increase accuracy (more simulations):**

```python
num_simulations = 1000  # More paths = smoother distribution
```

**To forecast further ahead:**

```python
num_days = 20  # 4 weeks (20 trading days)
```

---

## Understanding the Output

### Console Output

```
Mean final price after 5 days: $123.25
90% confidence interval: [$114.65, $133.42]
```

**Interpretation:**

- Average predicted price: $123.25
- 90% confidence: Price will likely be between $114.65 and $133.42
- 10% chance it falls outside this range (5% above, 5% below)

### Visualization Components

1. **Light Blue Lines**: Individual simulation paths showing different possible outcomes
2. **Red Line**: Average of all simulations - the "expected" path
3. **Green Dashed Line**: Starting price for reference
4. **Orange Dashed Lines**: Confidence interval boundaries
5. **Text Box**: Quick reference statistics

---

## Usage Instructions

### Basic Usage

1. Run the entire script
2. Wait for data download (a few seconds)
3. View the printed statistics
4. Examine the visualization

### Experimenting with Different Stocks

```python
ticker = 'TSLA'  # High volatility stock
# vs
ticker = 'JNJ'   # Low volatility stock
```

Compare how the prediction ranges differ!

### Testing Different Timeframes

```python
# Short-term (1 week)
num_days = 5

# Medium-term (1 month)
num_days = 20

# Long-term (3 months)
num_days = 60
```

Notice how uncertainty grows with time.

---

## Validation Approach

The model was validated by:

1. **Backtesting**: Running simulations from historical dates
2. **Comparing predictions vs actual outcomes**: Checking if real prices fell within predicted ranges
3. **Results**: In 20 test cases, actual prices fell within the 90% confidence interval 100% of the time, suggesting the model conservatively captures uncertainty

### Why Not Exact Predictions?

**Important**: This model does NOT predict exact future prices. Stock markets are inherently uncertain due to:

- Unpredictable news and events
- Market sentiment shifts
- Economic changes
- Company-specific developments

**What it DOES provide**: A realistic range of possibilities based on historical behavior.

---

## Limitations

### Model Assumptions

1. **Normal Distribution**: Assumes returns follow a bell curve (reality has "fatter tails" - extreme events are more common than normal distribution predicts)
2. **Constant Volatility**: Assumes volatility stays the same (reality: volatility changes over time)
3. **No Trends**: Doesn't account for market trends, earnings reports, or news
4. **Historical Patterns**: Assumes past behavior predicts future patterns (not always true)

### What This Model Doesn't Include

- Market sentiment analysis
- Company fundamentals (earnings, revenue)
- Macroeconomic factors (interest rates, inflation)
- Sector-specific trends
- Black swan events

### When Not to Use This Model

- For actual trading decisions without additional analysis
- During highly volatile market conditions
- Around major company announcements or earnings
- For stocks with limited historical data

---

## Future Improvements

This project represents the foundation of Monte Carlo simulation for stock prices. **I will continue to develop and enhance this model** with the following planned improvements:

### Short-term Enhancements

- [.] Add multiple stock comparison functionality
- [.] Implement different timeframe presets (1 week, 1 month, 3 months)
- [.] Calculate additional risk metrics (Value at Risk, Sharpe Ratio)
- [.] Add probability of profit/loss calculations
- [ ] Create interactive parameter inputs

### Medium-term Goals

- [ ] Implement more sophisticated models (GARCH for time-varying volatility)
- [ ] Add drift adjustment based on market trends
- [ ] Include correlation analysis for portfolio simulation
- [ ] Develop a web-based dashboard interface
- [ ] Add real-time data fetching capabilities

### Long-term Vision

- [ ] Incorporate machine learning for volatility prediction
- [ ] Add sentiment analysis from news/social media
- [ ] Build portfolio optimization tools
- [ ] Implement options pricing models
- [ ] Create backtesting framework with performance metrics

### Research Areas to Explore

- Jump diffusion models (for capturing sudden price movements)
- Stochastic volatility models
- Fat-tailed distributions (to better capture extreme events)
- Integration with fundamental analysis

---

## Conclusion

This Monte Carlo simulation demonstrates how randomness and statistics can model financial uncertainty. While not a crystal ball, it provides a framework for understanding the range of possible outcomes and helps develop intuition about risk in financial markets.

**Key Takeaway**: The goal is not to predict the future with certainty, but to quantify uncertainty and make informed decisions despite it.

---

_This is an educational project demonstrating Monte Carlo methods. It should not be used as the sole basis for investment decisions. Always conduct thorough research and consider consulting financial professionals before making investment choices._

**Project Status**: Active Development  
**Last Updated**: December 2025  
**Next Milestone**: Adding portfolio simulation capabilities
