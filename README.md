# Monte Carlo Stock Price Simulator & Validation Dashboard

## Project Overview

This project is a **data science–oriented Monte Carlo simulation system** for analyzing uncertainty in future stock prices.
Instead of predicting a single value, the model generates a **distribution of possible outcomes**, helping reason about risk, variability, and uncertainty.

The focus of this project is on:

- probabilistic modeling
- time series analysis
- model validation
- clean and maintainable Python code

This project is designed for someone **learning data science**, not for high-frequency trading or advanced quantitative finance research.

---

## Project Goals

- Model uncertainty in stock prices using simulation
- Understand how volatility affects future outcomes
- Validate model behavior using historical data
- Build an interactive interface for experimentation
- Practice an end-to-end data science workflow

---

## Key Features

### Monte Carlo Simulation

- Simulates thousands of future price paths
- Uses log returns and Geometric Brownian Motion (GBM)
- Supports multiple forecast horizons (days to one year)
- Produces realistic, non-negative price paths

### Risk and Distribution Metrics

- 90 percent confidence intervals
- Value at Risk (VaR) at 95 percent and 99 percent levels
- Probability of gain versus loss
- Expected volatility and return estimates
- Illustrative Sharpe Ratio for comparison

### Historical Validation

- Rolling historical backtests
- Comparison of predicted ranges versus actual outcomes
- Coverage analysis for confidence intervals
- Visualization of model behavior over time

### Interactive Web Application

- Built using Streamlit
- User-selectable ticker and forecast horizon
- Adjustable number of simulations
- Separate pages for simulation and validation

---

## What This Project Demonstrates

### Data Science Skills

- Probabilistic modeling
- Time series data handling
- Feature engineering using log returns
- Model validation and diagnostics
- Interpretation of uncertainty

### Statistical Thinking

- Working with distributions instead of point predictions
- Use of percentiles and confidence intervals
- Awareness of modeling assumptions
- Avoidance of overconfidence in predictions

### Software Engineering

- Modular Python code
- Clear function boundaries
- Defensive error handling
- Version control and refactoring
- User-focused design

---

## Project Structure

```
Monte-Carlo-Simulation/
├── app.py
├── Monte_Carlo.py
├── pages/
│   └── Validation.py
├── requirements.txt
└── README.md
```

---

## Technology Stack

- Python
- NumPy
- Pandas
- Matplotlib
- Streamlit
- yfinance

---

## Modeling Approach

The simulation uses **Geometric Brownian Motion (GBM)** with log returns.

Plain-text representation of the model:

```
Price(t+1) = Price(t) * exp((mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * epsilon)
```

Where:

- mu is the mean of log returns
- sigma is the volatility of log returns
- epsilon is a random value drawn from a standard normal distribution

This approach:

- keeps prices positive
- reflects multiplicative growth
- is commonly used for financial time series modeling

---

## Validation Methodology

The model is evaluated using **historical backtesting**.

For each test period:

- Parameters are estimated using past data only
- Future price paths are simulated
- The realized price is compared against predicted confidence intervals

The goal is to **evaluate calibration**, not to claim predictive accuracy.

---

## Interpretation of Results

- Confidence intervals describe uncertainty, not guarantees
- Coverage rates are descriptive, not proof of correctness
- Model performance degrades during high-volatility periods
- Results should be interpreted alongside assumptions

---

## Assumptions and Limitations

### Assumptions

- Log returns are approximately normally distributed
- Volatility remains constant during the forecast period
- Historical data contains useful information

### Limitations

- No regime switching or volatility clustering
- No macroeconomic or fundamental features
- Reduced reliability during extreme market events

This project is **educational** and should not be used as financial advice.

---

## Installation

```
git clone https://github.com/XCODESSS/Monte-Carlo-Simulation.git
cd Monte-Carlo-Simulation
pip install -r requirements.txt
```

---

## Running the Application

```
streamlit run app.py
```

Navigate between:

- Simulation page
- Validation page

---

## Future Improvements

- Time-varying volatility models (GARCH, stochastic volatility)
- Portfolio-level simulations
- Exportable reports (PDF, Excel)
- Model comparison tools
- Additional distribution options (skewed distributions)

---

## Learning Outcomes

This project represents a shift from:

- single predictions to distributions
- model building to model evaluation
- coding to communication

It reflects practical data science thinking applied to financial data.

---

## Project Status

- Status: Active development
- Version: 2.0
- Focus: Validation, clarity, and usability
