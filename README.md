# Monte Carlo Stock Simulator

Probabilistic stock price forecasting using Geometric Brownian Motion. **90% validated accuracy** on historical backtests.

**[Live Demo](https://monte-carlo-simulation-1.streamlit.app/)** | **[Documentation](#how-it-works)**

---

## Results

| Metric                   | Value                                |
| ------------------------ | ------------------------------------ |
| Validation Accuracy      | 90.0% (30+ backtests)                |
| Performance              | 1000 simulations in <1 sec           |
| Statistical Significance | p-value 0.67 (within expected range) |

---

## Features

- **Monte Carlo Simulation** — Normal and Student-t (fat-tailed) distributions
- **Risk Metrics** — VaR (95%, 99%), Sharpe Ratio, confidence intervals
- **Validation Dashboard** — Automated backtesting with regime analysis
- **Model Comparison** — Side-by-side Normal vs Student-t performance

---

## Quick Start

```bash
git clone https://github.com/XCODESSS/Monte-Carlo-Simulation
cd Monte-Carlo-Simulation
pip install -r requirements.txt
streamlit run app.py
```

---

## How It Works

**Model:** Geometric Brownian Motion
**`S(t+1) = S(t) × exp((μ - 0.5σ²)Δt + σ√Δt × ε)`**

- `μ`, `σ` estimated from historical log returns
- `ε` ~ Normal(0,1) or Student-t(df) for fat tails
- 90% confidence intervals from simulation percentiles (5th, 95th)

**Why Student-t?** Real markets have fatter tails than Normal distribution predicts. Student-t captures extreme events (crashes, rallies) more accurately.

---

## Assumptions

1. **Constant volatility** — Model uses historical σ, but real volatility changes over time
2. **Log-normal returns** — Assumes returns follow GBM; ignores jumps, mean reversion
3. **No fundamental factors** — Doesn't consider earnings, P/E, macroeconomic data
4. **Historical parameters** — Future may not resemble past

---

## Where It Fails

| Condition                          | Behavior                                    |
| ---------------------------------- | ------------------------------------------- |
| High volatility regimes            | Hit rate drops ~5-10%                       |
| Regime changes (bull→bear)         | Model lags behind transitions               |
| Black swan events                  | Even Student-t underestimates extreme tails |
| Long forecast horizons (>3 months) | Intervals become very wide (50%+)           |

**Directional accuracy: ~50%** — The model quantifies uncertainty well but doesn't predict direction better than chance.

---

## When To Use This

**Good for:**

- Understanding price uncertainty ranges
- Risk assessment (VaR, downside scenarios)
- Comparing volatility across assets
- Educational purposes

**Not good for:**

- Trading signals
- Predicting direction
- Long-term forecasts (>3 months)

---

## Tech Stack

Python · NumPy · Pandas · SciPy · Streamlit · yfinance

---

## Contact

**Shreyarth** — [LinkedIn](https://linkedin.com/in/yourprofile) · [GitHub](https://github.com/XCODESSS)

Seeking Data Science / Financial Analysis internships for Summer 2026.

<div align="center">

### ⭐ If this helped you understand Monte Carlo simulations, please star the repo!

**Built with:** Precision · Validated with Data · Ready for Production

</div>
