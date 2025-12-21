# Stochastic Modeling: Monte Carlo Stock Price Simulation

## Project Overview

This project implements a **Monte Carlo Simulation** to model the stochastic trajectories of equity prices based on historical volatility and drift. By generating a large ensemble of potential price paths, this tool quantifies market uncertainty and provides a probabilistic distribution of future outcomes, moving beyond deterministic "single-point" forecasting.

**Author:** Shreyarth Thakor  
**Academic Context:** IT Undergraduate (3rd Semester)  
**Project Status:** Active Development (Phase 1 Benchmarks Achieved)

---

## Methodology & Mathematical Foundation

Monte Carlo methods utilize repeated random sampling to resolve complex probabilistic systems. This implementation assumes that stock prices follow a **Geometric Brownian Motion (GBM)** process.

### 1. Data Acquisition & Normalization

The model ingests historical price data via the `yfinance` API. It calculates **Daily Arithmetic Returns** to determine the statistical properties of the asset:

$$R_t = \frac{P_t - P_{t-1}}{P_{t-1}}$$

### 2. Stochastic Modeling

Each simulated price step is calculated by applying a random shock to the drift of the asset:

$$P_{t+1} = P_t \times (1 + r_{random})$$

Where the random return ($r_{random}$) is defined as:

$$r_{random} = \mu + (\sigma \cdot \epsilon)$$

- **$\mu$ (Drift):** The average historical daily return, representing the asset's expected directional movement.
- **$\sigma$ (Volatility):** The standard deviation of daily returns, representing market risk.
- **$\epsilon$:** A random variable drawn from a **Standard Normal Distribution** $N(0, 1)$.

---

## Core Features & Analytical Engine

The simulation is built on a modular Python data pipeline that now includes advanced financial risk metrics:

- **Simulation Core:** A nested execution framework where the outer loop generates independent stochastic paths and the inner loop iterates through the forecast horizon.
- **Statistical Derivatives:** Post-simulation derivation of the 5th and 95th percentiles to establish a **90% Confidence Interval**.
- **Risk Analytics (Completed):** \* **Value at Risk (VaR):** Quantification of the potential maximum loss at a 95% confidence level.
  - **Sharpe Ratio:** Calculation of risk-adjusted returns based on simulated volatility.
  - **Probability Analysis:** Automated calculation of the likelihood of the asset finishing "In-the-Money" (above the starting price).
- **Visualization Suite:** A `matplotlib` implementation rendering a comprehensive "Fan Chart" of all simulated trajectories.

---

## Key Parameters & Configuration

| Parameter            | Identifier        | Default  | Description                                                 |
| :------------------- | :---------------- | :------- | :---------------------------------------------------------- |
| **Ticker Symbol**    | `ticker`          | `AAPL`   | The equity identifier for data retrieval.                   |
| **Lookback Period**  | `start`/`end`     | 6 Months | The historical window used to calculate $\mu$ and $\sigma$. |
| **Simulation Depth** | `num_simulations` | 100      | The total number of independent stochastic paths.           |
| **Forecast Horizon** | `num_days`        | 5        | The number of trading days to project forward.              |

---

## Critical Assumptions & Limitations

While Monte Carlo simulations are a cornerstone of risk management, this model operates under the following financial assumptions:

1. **Gaussian Distribution:** It assumes market returns follow a normal distribution. In reality, markets often exhibit **Kurtosis** (Fat Tails), where extreme events occur more frequently than predicted.
2. **Stationarity:** The model assumes that historical volatility and drift will remain constant over the forecast horizon.
3. **Excluded Exogenous Variables:** The current iteration does not account for macroeconomic shifts, corporate earnings reports, or "Black Swan" events.

---

## Future Development Roadmap

### Phase 2: User Interface & Refinement (Planned Q1 2026)

- [ ] **Interactive Dashboard:** Transition from a script-based execution to a web interface using **Streamlit** or **Flask**.
- [ ] **Parameter Dynamicism:** Allow users to input custom drift or volatility overrides to "Stress Test" specific market scenarios.
- [ ] **Portfolio Simulation:** Support for a basket of stocks accounting for inter-stock correlation.

### Phase 3: Advanced Computational Models (Planned Q2 2026)

- [ ] **GARCH Integration:** Implement Generalized Autoregressive Conditional Heteroskedasticity to model time-varying volatility.
- [ ] **Jump-Diffusion:** Incorporate Poisson processes to simulate sudden market "jumps" or crashes.
- [ ] **Backtesting Framework:** Evaluate model accuracy by comparing past simulations against actual historical results.

---

## Conclusion

This framework demonstrates the power of computational statistics in navigating financial uncertainty. Rather than predicting the future with certainty, it provides a disciplined lens through which risk can be measured and mitigated.

> **Disclaimer:** This is an educational tool for demonstrating computational methods. It does not constitute financial advice.

**Last Updated:** December 2025
