"""
Monte Carlo Stock Price Simulator - Streamlit App
Main interface for running single or comparative Monte Carlo simulations.
"""

# ── Imports
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os

from Monte_Carlo import (
    TIMEFRAMES,
    download_stock_data,
    calculate_statistics,
    run_monte_carlo,
    run_monte_carlo_student_t,
    calculate_metrics,
    plot_simulation,
)


# ── Cached Data Loader 
def cached_download_stock_data(ticker, start_date, end_date):
    """Cached wrapper for download_stock_data"""
    return download_stock_data(ticker, start_date, end_date)


# ── Page Config & Global Styles 
st.set_page_config(layout="wide", page_title="Monte Carlo Stock Simulator")

plt.rcParams.update(
    {
        "figure.facecolor": "#0e1117",
        "axes.facecolor": "#0e1117",
        "savefig.facecolor": "#0e1117",
        "text.color": "white",
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "axes.edgecolor": "#2d2d2d",
        "grid.color": "#2d2d2d",
    }
)
sns.set_style(
    "darkgrid",
    {"axes.facecolor": "#0e1117", "figure.facecolor": "#0e1117"},
)

# ── Chart Style Constants 
BLUE = "#4A90E2"
RED = "#E74C3C"
GOLD = "#FFD700"
BG = "#0e1117"
GRID = "#2d2d2d"
PANEL_BG = "#1e1e1e"


# ── Helper: style a matplotlib axis for dark theme
def style_axis(ax):
    """Apply consistent dark-theme styling to a matplotlib axis."""
    ax.set_facecolor(BG)
    ax.grid(True, alpha=0.3, color=GRID)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)


# ── Helper: build a single-model price-path chart
def plot_price_paths(ax, days_array, simulations, color, title, starting_price):
    """Plot simulation paths, percentile bands, and median on an axis."""
    num_paths = min(200, len(simulations))
    ax.plot(
        days_array,
        simulations[:num_paths].T,
        color=color,
        alpha=0.05,
        linewidth=0.5,
        zorder=1,
    )

    p5, p50, p95 = np.percentile(simulations, [5, 50, 95], axis=0)
    ax.fill_between(
        days_array, p5, p95, color=color, alpha=0.3, label="90% CI", zorder=2
    )
    ax.plot(
        days_array, p50, color="#FFFFFF", linewidth=3, label="Median", zorder=4
    )
    ax.axhline(
        y=starting_price,
        color=GOLD,
        linestyle="--",
        linewidth=2,
        alpha=0.8,
        label="Start Price",
        zorder=3,
    )

    ax.set_title(
        title, fontsize=14, fontweight="bold", pad=10, color="white"
    )
    ax.set_ylabel("Price ($)", fontsize=11, fontweight="bold", color="white")
    ax.set_xlabel("Trading Days", fontsize=11, fontweight="bold", color="white")
    ax.legend(
        loc="upper left",
        fontsize=9,
        facecolor=PANEL_BG,
        edgecolor=GRID,
        framealpha=0.95,
        labelcolor="white",
    )
    style_axis(ax)


#  SIDEBAR – User Configuration
st.title("Monte Carlo Stock Price Simulator")

with st.sidebar:
    st.header("Configuration")

    # ── Ticker & API Key
    ticker = st.text_input("Stock Ticker", value="AAPL")

    alpha_vantage_key = None
    try:
        if "ALPHA_VANTAGE_KEY" in st.secrets:
            alpha_vantage_key = st.secrets["ALPHA_VANTAGE_KEY"]
    except FileNotFoundError:
        pass
    if not alpha_vantage_key and "ALPHA_VANTAGE_KEY" in os.environ:
        alpha_vantage_key = os.environ["ALPHA_VANTAGE_KEY"]
    if ticker.endswith(".NS") and not alpha_vantage_key:
        alpha_vantage_key = st.text_input(
            "Alpha Vantage API Key",
            type="password",
            help="Required for Indian stocks (.NS)",
        )
        if not alpha_vantage_key:
            st.warning("⚠️ Alpha Vantage API key required for Indian stocks.")

    # ── Forecast Period
    st.subheader("📅 Forecast Period")
    use_custom_days = st.checkbox("Use custom number of days")

    if use_custom_days:
        num_days = st.number_input(
            "Number of Trading Days", min_value=1, max_value=365, value=20
        )
        timeframe_label = f"{num_days} days"
    else:
        timeframe = st.selectbox(
            "Forecast Timeframe", list(TIMEFRAMES.keys())
        )
        num_days = TIMEFRAMES[timeframe]
        timeframe_label = timeframe.replace("_", " ")

    # ── Distribution Model
    st.subheader("📊 Distribution Model")
    compare_models = st.checkbox(
        "🔀 Compare Normal vs Student-t Models",
        value=False,
        help="Run both models side-by-side for comparison",
    )

    if compare_models:
        df_val = st.slider(
            "Degrees of Freedom (Student-t)",
            min_value=3,
            max_value=30,
            value=7,
            help="Lower values = fatter tails",
        )
        distribution = "Both"
    else:
        distribution = st.selectbox(
            "Return Distribution", ["Normal", "Student-t (Fat Tails)"]
        )
        df_val = None
        if distribution == "Student-t (Fat Tails)":
            df_val = st.slider(
                "Degrees of Freedom",
                min_value=3,
                max_value=30,
                value=7,
                help="Lower values = fatter tails",
            )

    # ── Simulation Count
    st.subheader("🔢 Simulations")
    num_simulations = st.slider(
        "Number of Simulations",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100,
    )

    st.divider()
    run_button = st.button(
        "🚀 Run Simulation", type="primary", use_container_width=True
    )


#  MAIN PANEL – Simulation Execution & Results
if run_button:

    # ── 1. Download Data
    with st.spinner("Downloading stock data..."):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 10)
        try:
            data = cached_download_stock_data(
                ticker,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
            )
            if data.empty:
                st.error(f"❌ No data found for ticker '{ticker}'.")
                st.stop()
        except Exception as e:
            st.error(f"❌ Error downloading data: {str(e)}")
            st.info(
                "💡 Tip: Use a valid stock ticker (e.g., AAPL, MSFT, GOOGL)"
            )
            st.stop()

    # ── 2. Calculate Statistics
    with st.spinner("Calculating statistics..."):
        stats = calculate_statistics(data)

    # ── 3. Run Simulations
    if distribution == "Both":
        with st.spinner(
            f"Running {num_simulations} simulations for both models..."
        ):
            simulations_normal = run_monte_carlo(
                stats["starting_price"],
                stats["mu"],
                stats["sigma"],
                num_days,
                num_simulations,
            )
            simulations_student_t = run_monte_carlo_student_t(
                stats["starting_price"],
                stats["mu"],
                stats["sigma"],
                num_days,
                num_simulations,
                df=df_val,
            )
        metrics_normal = calculate_metrics(
            simulations_normal,
            stats["starting_price"],
            stats["mu"],
            stats["sigma"],
            num_days,
        )
        metrics_student_t = calculate_metrics(
            simulations_student_t,
            stats["starting_price"],
            stats["mu"],
            stats["sigma"],
            num_days,
        )
        # Default to normal for top-level summary cards
        simulations = simulations_normal
        metrics = metrics_normal
    else:
        with st.spinner(f"Running {num_simulations} simulations..."):
            if distribution == "Normal":
                simulations = run_monte_carlo(
                    stats["starting_price"],
                    stats["mu"],
                    stats["sigma"],
                    num_days,
                    num_simulations,
                )
            else:
                simulations = run_monte_carlo_student_t(
                    stats["starting_price"],
                    stats["mu"],
                    stats["sigma"],
                    num_days,
                    num_simulations,
                    df=df_val,
                )
        metrics = calculate_metrics(
            simulations,
            stats["starting_price"],
            stats["mu"],
            stats["sigma"],
            num_days,
        )

    # ── 4. Top-Level Metrics
    st.subheader("Simulation Results")
    st.write(
        f"**Ticker:** {ticker} | **Period:** {timeframe_label} "
        f"| **Distribution:** {distribution}"
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Price", f"${stats['starting_price']:.2f}")
    with col2:
        chg = metrics["mean_final_price"] - stats["starting_price"]
        pct = chg / stats["starting_price"] * 100
        st.metric(
            "Expected Price",
            f"${metrics['mean_final_price']:.2f}",
            f"{chg:+.2f} ({pct:+.1f}%)",
        )
    with col3:
        st.metric(
            "Value at Risk (95%)",
            f"${metrics['var_95_loss']:.2f}",
            delta_color="inverse",
        )
    with col4:
        st.metric("Probability of Profit", f"{metrics['prob_profit']:.1f}%")

    # ── 5. Gain / Loss Summary
    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        if metrics["avg_loss"] < 0:
            st.write(f"📉 **Average Loss:** {metrics['avg_loss']*100:.2f}%")
        st.write(f"**Probability of Loss:** {metrics['prob_loss']:.1f}%")
    with col_b:
        if metrics["avg_gain"] > 0:
            st.write(f"📈 **Average Gain:** {metrics['avg_gain']*100:.2f}%")
        st.write(
            f"**Probability of Profit:** {metrics['prob_profit']:.1f}%"
        )

    # ── 6. Sharpe Ratio
    st.divider()
    st.subheader("Risk-Adjusted Performance")
    st.write(f"**Sharpe Ratio:** {metrics['sharpe_ratio_post']:.2f}")

    if metrics["sharpe_ratio_post"] > 2:
        st.success("Excellent risk-adjusted returns")
    elif metrics["sharpe_ratio_post"] > 1:
        st.info("Good risk-adjusted returns")
    else:
        st.warning("Low risk-adjusted returns")

    # ── 7. Visualization
    if distribution == "Both":
        # Side-by-side price path charts
        days_array = np.arange(1, num_days + 1)

        col1, col2 = st.columns(2)
        with col1:
            fig1, ax1 = plt.subplots(
                figsize=(8, 6), facecolor=BG, dpi=100
            )
            plot_price_paths(
                ax1,
                days_array,
                simulations_normal,
                BLUE,
                "Normal Distribution (GBM)",
                stats["starting_price"],
            )
            plt.tight_layout()
            st.pyplot(fig1, use_container_width=True)

        with col2:
            fig2, ax2 = plt.subplots(
                figsize=(8, 6), facecolor=BG, dpi=100
            )
            plot_price_paths(
                ax2,
                days_array,
                simulations_student_t,
                RED,
                f"Student-t Distribution (df={df_val})",
                stats["starting_price"],
            )
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)

        # -- Full-width distribution comparison histogram --
        st.subheader("📊 Distribution Comparison")
        fig3, ax3 = plt.subplots(
            figsize=(14, 5), facecolor=BG, dpi=100
        )

        final_norm = simulations_normal[:, -1]
        final_t = simulations_student_t[:, -1]

        sns.histplot(
            final_norm,
            bins=40,
            alpha=0.4,
            color=BLUE,
            label=(
                f"Normal (Mean: ${metrics_normal['mean_final_price']:.2f})"
            ),
            element="step",
            kde=True,
            ax=ax3,
            edgecolor=BLUE,
            linewidth=1.5,
        )
        sns.histplot(
            final_t,
            bins=40,
            alpha=0.4,
            color=RED,
            label=(
                f"Student-t "
                f"(Mean: ${metrics_student_t['mean_final_price']:.2f})"
            ),
            element="step",
            kde=True,
            ax=ax3,
            edgecolor=RED,
            linewidth=1.5,
        )

        for bound in (
            metrics_normal["lower_bound"],
            metrics_normal["upper_bound"],
        ):
            ax3.axvline(
                bound, color=BLUE, linestyle="--", linewidth=1.5, alpha=0.6
            )
        for bound in (
            metrics_student_t["lower_bound"],
            metrics_student_t["upper_bound"],
        ):
            ax3.axvline(
                bound, color=RED, linestyle="--", linewidth=1.5, alpha=0.6
            )

        ax3.set_xlabel(
            "Final Price ($)", fontsize=12, fontweight="bold", color="white"
        )
        ax3.set_ylabel(
            "Frequency", fontsize=12, fontweight="bold", color="white"
        )
        ax3.set_title(
            "Final Price Distribution Comparison",
            fontsize=13,
            fontweight="bold",
            pad=10,
            color="white",
        )
        ax3.legend(
            loc="upper right",
            fontsize=10,
            facecolor=PANEL_BG,
            edgecolor=GRID,
            framealpha=0.95,
            labelcolor="white",
        )
        ax3.grid(True, alpha=0.3, axis="y", color=GRID)
        style_axis(ax3)
        plt.tight_layout()
        st.pyplot(fig3, use_container_width=True)

        # -- Comparison metrics table --
        st.write("---")
        st.subheader("📊 Model Comparison Metrics")

        comp1, comp2, comp3, comp4 = st.columns(4)
        with comp1:
            st.metric(
                "Lower Bound (5%)",
                f"Normal: ${metrics_normal['lower_bound']:.2f}\n"
                f"Student-t: ${metrics_student_t['lower_bound']:.2f}",
            )
        with comp2:
            st.metric(
                "Upper Bound (95%)",
                f"Normal: ${metrics_normal['upper_bound']:.2f}\n"
                f"Student-t: ${metrics_student_t['upper_bound']:.2f}",
            )
        with comp3:
            w_n = (
                (
                    metrics_normal["upper_bound"]
                    - metrics_normal["lower_bound"]
                )
                / stats["starting_price"]
                * 100
            )
            w_t = (
                (
                    metrics_student_t["upper_bound"]
                    - metrics_student_t["lower_bound"]
                )
                / stats["starting_price"]
                * 100
            )
            st.metric(
                "Interval Width",
                f"Normal: {w_n:.1f}%\nStudent-t: {w_t:.1f}%",
            )
        with comp4:
            st.metric(
                "VaR (95%)",
                f"Normal: ${metrics_normal['var_95_loss']:.2f}\n"
                f"Student-t: ${metrics_student_t['var_95_loss']:.2f}",
            )

        st.info(
            "💡 **Key Insight**: Student-t distribution typically shows "
            "wider confidence intervals due to fat tails, capturing more "
            "extreme events but being more conservative."
        )

    else:
        # -- Single-model visualization (uses Monte_Carlo.py's plot) --
        fig = plot_simulation(
            simulations,
            metrics,
            stats["starting_price"],
            stats["sigma"],
            ticker,
            num_days,
            num_simulations,
        )
        st.pyplot(fig, use_container_width=True)