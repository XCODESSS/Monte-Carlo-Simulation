# Monte Carlo Stock Simulator

### Probabilistic forecasting with 90% validated accuracy

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-red.svg)](https://your-app.streamlit.app)
[![Validation](https://img.shields.io/badge/Accuracy-90%25-success.svg)]()

**[ Try Live Demo](https://monte-carlo-simulation-1.streamlit.app/)** | **[📧 Contact Me](#contact)**

<div align="center">
<img src="screenshots/main-demo.gif" alt="Demo" width="800"/>
</div>

---

## What Makes This Different

Most Monte Carlo projects are toy examples. This one actually works:

**90% hit rate** on 30+ historical backtests  
 **Fat-tailed distributions** (Student-t) for realistic crash modeling  
 **Side-by-side comparison** of Normal vs Student-t distributions  
 **Statistical validation** with p-value testing and diagnostics  
 **Production-ready code** with 30x performance optimization

---

## Key Results

| What I Built                    | Why It Matters                                      |
| ------------------------------- | --------------------------------------------------- |
| **90% Validation Accuracy**     | Predicted ranges captured actual prices 9/10 times  |
| **1000+ Simulations in <1 sec** | Vectorized NumPy (30x faster than loops)            |
| **Student-t Fat Tails**         | Captures market crashes Normal distribution misses  |
| **Real-time Web App**           | Interactive Streamlit dashboard, Bloomberg-style UI |

---

## Tech Stack

**Core:** Python · NumPy · Pandas · SciPy  
**Modeling:** Geometric Brownian Motion · Student-t Distribution · Log Returns  
**Validation:** Historical Backtesting · Statistical Significance Testing  
**Frontend:** Streamlit · Matplotlib · Seaborn  
**Data:** yfinance API

---

## Features

### Monte Carlo Engine

<img src="screenshots/simulation.png" width="400"/>

- 1000+ price paths using Geometric Brownian Motion
- Normal vs Student-t distribution comparison
- Risk metrics: VaR, Sharpe Ratio, confidence intervals

### Validation Dashboard

<img src="screenshots/validation.png" width="400"/>

- Automated historical backtesting
- 90% accuracy confirmed across 30+ periods
- Statistical diagnostics and p-value testing

---

## Quick Start

```bash
git clone https://github.com/XCODESSS/Monte-Carlo-Simulation
cd Monte-Carlo-Simulation
pip install -r requirements.txt
streamlit run app.py
```

**Try it live:** [https://monte-carlo-simulation-1.streamlit.app/](https://monte-carlo-simulation-1.streamlit.app/)

---

## What I Learned

**Technical Skills:**

- Statistical modeling and uncertainty quantification
- Time series analysis with financial data
- Performance optimization (vectorization)
- Model validation methodology
- Production-grade Python architecture

**Domain Knowledge:**

- Financial modeling (GBM, VaR, Sharpe Ratio)
- Fat-tailed distributions for extreme events
- Backtesting best practices (avoiding data leakage)
- Risk management principles

**Evolution:** Started as a coin-flip simulator → Added risk metrics → Implemented rigorous validation → Achieved 90% accuracy → Built production UI

---

## Technical Highlights

**Geometric Brownian Motion Formula:**

```
S(t+1) = S(t) × exp((μ - 0.5σ²)Δt + σ√Δt × ε)
where ε ~ N(0,1) or Student-t(df)
```

**Why Student-t Distribution?**  
Real markets have more extreme events than Normal distribution predicts. Student-t with df=5 captures these "Black Swan" events realistically.

**Performance Optimization:**  
Vectorized NumPy operations instead of loops → 30x speedup (5s → 0.15s for 1000 simulations)

---

## Project Status

**Version:** 2.5 (Production)

**Validation Results:**

- 90.0% hit rate (expected: 90% ± 5.7%)
- P-value: 0.67 (not significantly different from expected)
- Tested across bull markets, bear markets, and COVID crash

---

## Contact

**Shreyarth** · 4th Semester IT Student

**Email:** your.email@example.com  
 **LinkedIn:** [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)  
 **GitHub:** [@XCODESSS](https://github.com/XCODESSS)

**Seeking:** Data Science Internships | Financial Analysis Roles

---

<div align="center">

### ⭐ If this helped you understand Monte Carlo simulations, please star the repo!

**Built with:** Precision · Validated with Data · Ready for Production

</div>
