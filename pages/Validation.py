"""
Model Validation Dashboard
Backtests the Monte Carlo model on historical data to verify calibration accuracy.
"""


# IMPORTS

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy.stats import binomtest
import os

from Monte_Carlo import (
    download_stock_data,
    calculate_statistics,
    run_monte_carlo,
    run_monte_carlo_student_t,
    calculate_metrics,
    TIMEFRAMES,
)
from Calibration import (
    calculate_calibration_metrics,
    calculate_directional_accuracy,
    identify_failure_patterns,
    calculate_rolling_coverage,
)


# STYLING


plt.rcParams["figure.facecolor"] = "#0e1117"
plt.rcParams["axes.facecolor"] = "#0e1117"
plt.rcParams["savefig.facecolor"] = "#0e1117"
plt.rcParams["text.color"] = "white"
plt.rcParams["axes.labelcolor"] = "white"
plt.rcParams["xtick.color"] = "white"
plt.rcParams["ytick.color"] = "white"
plt.rcParams["axes.edgecolor"] = "#2d2d2d"
plt.rcParams["grid.color"] = "#2d2d2d"
sns.set_style(
    "darkgrid",
    {"axes.facecolor": "#0e1117", "figure.facecolor": "#0e1117"},
)


# HELPER FUNCTIONS



def cached_download_stock_data(ticker, start_date, end_date):
    """Cached wrapper for download_stock_data"""
    return download_stock_data(ticker, start_date, end_date)


def run_single_model_simulation(stats, num_days, distribution, df_val):
    """Run simulation for a single distribution and return bounds"""
    if distribution == "Student-t (Fat Tails)":
        sims = run_monte_carlo_student_t(
            stats["starting_price"],
            stats["mu"],
            stats["sigma"],
            num_days,
            1000,
            df=df_val,
        )
    else:
        sims = run_monte_carlo(
            stats["starting_price"],
            stats["mu"],
            stats["sigma"],
            num_days,
            1000,
        )
    m = calculate_metrics(
        sims,
        stats["starting_price"],
        stats["mu"],
        stats["sigma"],
        num_days,
    )
    return m["lower_bound"], m["upper_bound"]


def run_compare_simulation(stats, num_days, df_val):
    """Run both Normal and Student-t simulations, return both bounds"""
    sims_n = run_monte_carlo(
        stats["starting_price"],
        stats["mu"],
        stats["sigma"],
        num_days,
        1000,
    )
    sims_t = run_monte_carlo_student_t(
        stats["starting_price"],
        stats["mu"],
        stats["sigma"],
        num_days,
        1000,
        df=df_val,
    )
    m_n = calculate_metrics(
        sims_n,
        stats["starting_price"],
        stats["mu"],
        stats["sigma"],
        num_days,
    )
    m_t = calculate_metrics(
        sims_t,
        stats["starting_price"],
        stats["mu"],
        stats["sigma"],
        num_days,
    )
    return (
        m_n["lower_bound"],
        m_n["upper_bound"],
        m_t["lower_bound"],
        m_t["upper_bound"],
    )


def build_single_result(
    test_date, forecast_date, stats, actual_price, lower, upper, distribution
):
    """Build a result dict for a single-distribution backtest"""
    if actual_price is not None:
        within = lower <= actual_price <= upper
    else:
        within = False
    return {
        "Test Date": test_date.strftime("%Y-%m-%d"),
        "Forecast Date": (
            forecast_date.strftime("%Y-%m-%d") if forecast_date else "N/A"
        ),
        "Starting Price": stats["starting_price"],
        "Actual Price": actual_price,
        "mu": stats["mu"],
        "Lower Bound": lower,
        "Upper Bound": upper,
        "Within Bounds": within,
        "Volatility": stats["sigma"] * np.sqrt(252) * 100,
        "Distribution": distribution,
    }


def build_compare_result(
    test_date, forecast_date, stats, actual_price, lb_n, ub_n, lb_t, ub_t
):
    """Build a result dict for a compare-both backtest"""
    if actual_price is not None:
        within_n = lb_n <= actual_price <= ub_n
        within_t = lb_t <= actual_price <= ub_t
    else:
        within_n = False
        within_t = False
    return {
        "Test Date": test_date.strftime("%Y-%m-%d"),
        "Forecast Date": (
            forecast_date.strftime("%Y-%m-%d") if forecast_date else "N/A"
        ),
        "Starting Price": stats["starting_price"],
        "Actual Price": actual_price,
        "mu": stats["mu"],
        "Lower Bound Normal": lb_n,
        "Upper Bound Normal": ub_n,
        "Lower Bound Student-t": lb_t,
        "Upper Bound Student-t": ub_t,
        "Within Bounds Normal": within_n,
        "Within Bounds Student-t": within_t,
        "Volatility": stats["sigma"] * np.sqrt(252) * 100,
        "Distribution": "Both",
    }


def rename_for_calibration(df_valid, bounds_prefix):
    """Rename bound columns so calibration functions can consume them"""
    base_cols = [
        "Test Date",
        "Starting Price",
        "Actual Price",
        "Volatility",
        "mu",
    ]
    return df_valid.rename(
        columns={
            f"Within Bounds {bounds_prefix}": "Within Bounds",
            f"Lower Bound {bounds_prefix}": "Lower Bound",
            f"Upper Bound {bounds_prefix}": "Upper Bound",
        }
    )[base_cols + ["Within Bounds", "Lower Bound", "Upper Bound"]]


def render_regime_table(calib_metrics):
    """Display volatility regime analysis table"""
    regime_data = []
    for regime, s in calib_metrics["regime_analysis"].items():
        regime_data.append(
            {
                "Regime": regime,
                "Hit Rate": f"{s['hit_rate']:.1f}%",
                "Tests": s["count"],
                "Avg Miss Magnitude": f"{s['avg_miss_magnitude']:.1f}%",
            }
        )
    if regime_data:
        st.dataframe(
            pd.DataFrame(regime_data),
            use_container_width=True,
            hide_index=True,
        )
        low = calib_metrics["regime_analysis"].get("Low", {}).get(
            "hit_rate", 0
        )
        high = calib_metrics["regime_analysis"].get("High", {}).get(
            "hit_rate", 0
        )
        if low > 0 and high > 0:
            drop = low - high
            st.info(
                f"💡 **Key Finding:** Model accuracy drops {drop:.1f}% "
                "in high volatility regimes"
            )


def render_directional_accuracy(dir_accuracy):
    """Display directional accuracy metrics"""
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            "Overall Direction",
            f"{dir_accuracy['overall_accuracy']*100:.1f}%",
            help="% of times we correctly predicted up vs down",
        )
    with c2:
        st.metric(
            "Upward Moves",
            f"{dir_accuracy['upward_accuracy']*100:.1f}%",
            help="Accuracy when stock went up",
        )
    with c3:
        st.metric(
            "Downward Moves",
            f"{dir_accuracy['downward_accuracy']*100:.1f}%",
            help="Accuracy when stock went down",
        )


def render_failure_analysis(failure_patterns, n_valid):
    """Display failure case analysis"""
    ca, cb = st.columns(2)
    with ca:
        st.metric(
            "Major Failures (>15% miss)",
            int(failure_patterns["major_failures"]),
            delta=f"{failure_patterns['failure_rate']:.1f}% of all tests",
        )
    with cb:
        st.metric(
            "Total Misses",
            int(failure_patterns["total_misses"]),
            delta=f"{failure_patterns['total_misses']/n_valid*100:.1f}% of tests",
        )
    if failure_patterns["patterns"]:
        st.write("**Failure Patterns:**")
        for p in failure_patterns["patterns"]:
            st.write(f"- {p}")


def render_worst_misses(worst_misses):
    """Display worst prediction misses table"""
    if not worst_misses:
        return
    worst_df = pd.DataFrame(worst_misses)
    worst_df["Actual Price"] = worst_df["Actual Price"].apply(
        lambda x: f"${x:.2f}"
    )
    worst_df["Lower Bound"] = worst_df["Lower Bound"].apply(
        lambda x: f"${x:.2f}"
    )
    worst_df["Upper Bound"] = worst_df["Upper Bound"].apply(
        lambda x: f"${x:.2f}"
    )
    worst_df["Miss Magnitude"] = worst_df["Miss Magnitude"].apply(
        lambda x: f"{x:.1f}%"
    )
    worst_df["Volatility"] = worst_df["Volatility"].apply(
        lambda x: f"{x:.1f}%"
    )
    st.dataframe(worst_df, use_container_width=True, hide_index=True)
    st.caption("*These represent the most significant prediction errors*")


def style_validation_ax(ax):
    """Apply consistent dark styling to a matplotlib axes"""
    ax.set_facecolor("#0e1117")
    ax.grid(True, alpha=0.3, color="#2d2d2d")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#2d2d2d")


def plot_validation_subplot(ax, dates, df, lower_col, upper_col, within_col, color, title):
    """Plot one validation subplot (bounds + hits/misses)"""
    style_validation_ax(ax)
    ax.fill_between(
        dates,
        df[lower_col],
        df[upper_col],
        alpha=0.3,
        color=color,
        label="90% Confidence Interval",
        zorder=1,
    )
    ax.plot(
        dates, df[lower_col], color=color, linestyle="--",
        linewidth=2, alpha=0.8, label="Bounds", zorder=2,
    )
    ax.plot(
        dates, df[upper_col], color=color, linestyle="--",
        linewidth=2, alpha=0.8, zorder=2,
    )
    hits = df[df[within_col] == True]
    misses = df[df[within_col] == False]
    if len(hits) > 0:
        ax.scatter(
            pd.to_datetime(hits["Test Date"]),
            hits["Actual Price"],
            color="#27AE60", s=80, zorder=5,
            label="✓ Within Bounds",
            marker="o", edgecolors="white", linewidths=1.5, alpha=0.8,
        )
    if len(misses) > 0:
        ax.scatter(
            pd.to_datetime(misses["Test Date"]),
            misses["Actual Price"],
            color="#E74C3C", s=120, zorder=6,
            label="✗ Outside Bounds",
            marker="X", edgecolors="white", linewidths=2, alpha=0.9,
        )
    ax.set_title(
        title, fontsize=14, fontweight="bold", pad=15, color="white"
    )
    ax.legend(
        loc="upper left", fontsize=9, facecolor="#1e1e1e",
        edgecolor="#2d2d2d", framealpha=0.95, labelcolor="white",
    )



# PAGE HEADER & USER INPUTS


st.title("📊 Model Validation Dashboard")
st.write(
    "Backtest the Monte Carlo model on historical data to verify accuracy"
)

st.subheader("Configuration")
col1, col2 = st.columns(2)

with col1:
    ticker = st.text_input("Stock Ticker", value="AAPL")

    # API Key for Indian Stocks
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

    timeframe = st.selectbox("Forecast Timeframe", list(TIMEFRAMES.keys()))
    num_backtests = st.slider(
        "Number of Backtests", min_value=10, max_value=100, value=30, step=5
    )

    st.write("---")
    distribution = st.selectbox(
        "Distribution Model",
        ["Normal", "Student-t (Fat Tails)", "Compare Both"],
        help="Choose which distribution to validate",
    )
    if distribution in ("Student-t (Fat Tails)", "Compare Both"):
        df_param = st.slider(
            "Degrees of Freedom (Student-t)",
            min_value=3, max_value=30, value=7,
            help="Lower values = fatter tails (more extreme events)",
        )
    else:
        df_param = None

with col2:
    backtest_end = st.date_input(
        "Backtest End Date",
        value=pd.Timestamp("2023-01-01"),
        help="Latest date to run backtest from. For COVID-19 testing, use 2020-02-15",
    )
    historical_start = st.date_input(
        "Historical Data Start",
        value=pd.Timestamp("2015-01-01"),
        help="How far back to use for parameter estimation",
    )

    st.write("---")
    use_rolling_window = st.checkbox(
        "Use Rolling Window",
        value=True,
        help="Use only recent data for parameter estimation",
    )
    if use_rolling_window:
        lookback_days = st.number_input(
            "Lookback Period (Days)",
            min_value=30, max_value=252 * 5, value=252, step=10,
            help="Number of trading days for calculating volatility and drift",
        )
    else:
        lookback_days = None

# INPUT VALIDATION


if historical_start >= backtest_end:
    st.error("❌ Historical Data Start must be before Backtest End Date")
    st.stop()


# BACKTEST EXECUTION


if st.button("Run Backtest", type="primary"):
    num_days = TIMEFRAMES[timeframe]
    end_date = pd.Timestamp(backtest_end)
    start_date_str = historical_start.strftime("%Y-%m-%d")
    end_date_str = backtest_end.strftime("%Y-%m-%d")

    test_dates = pd.date_range(
        end=end_date, periods=num_backtests, freq="ME"
    ).sort_values()

    # --- Validate ticker ---
    try:
        test_data = cached_download_stock_data(
            ticker, start_date_str, end_date_str
        )
        if test_data.empty:
            st.error(
                f"❌ No data found for ticker '{ticker}'. "
                "Please check the symbol."
            )
            st.stop()
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("💡 Tip: Make sure you're using a valid stock ticker")
        st.stop()

    # --- Run backtests ---
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    full_data = cached_download_stock_data(
        ticker, start_date_str, end_date_str
    )

    for i, test_date in enumerate(test_dates):
        status_text.text(f"Running backtest {i+1}/{num_backtests}...")
        progress_bar.progress((i + 1) / num_backtests)

        data = full_data[full_data.index <= test_date].copy()

        try:
            if len(data) < 30:
                continue

            # Rolling window filter
            if use_rolling_window and lookback_days and len(data) > lookback_days:
                stats_data = data.tail(lookback_days)
            else:
                stats_data = data

            stats = calculate_statistics(stats_data)

            # Get actual future price
            future_data = full_data[full_data.index > test_date]
            if len(future_data) >= num_days:
                actual_price = float(future_data["Close"].iloc[num_days - 1])
                actual_forecast_date = future_data.index[num_days - 1]
            else:
                actual_price = None
                actual_forecast_date = None

            # Run simulation and store result
            if distribution == "Compare Both":
                lb_n, ub_n, lb_t, ub_t = run_compare_simulation(
                    stats, num_days, df_param
                )
                results.append(
                    build_compare_result(
                        test_date, actual_forecast_date, stats,
                        actual_price, lb_n, ub_n, lb_t, ub_t,
                    )
                )
            else:
                lower, upper = run_single_model_simulation(
                    stats, num_days, distribution, df_param
                )
                results.append(
                    build_single_result(
                        test_date, actual_forecast_date, stats,
                        actual_price, lower, upper, distribution,
                    )
                )

        except Exception as e:
            st.error(f"Error at test {i+1} ({test_date}): {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            continue

    progress_bar.empty()
    status_text.empty()


    # RESULTS PROCESSING


    if not results:
        st.error(
            "No valid backtest results. Try different parameters or date range."
        )
        st.stop()

    df_results = pd.DataFrame(results)
    df_valid = df_results[df_results["Actual Price"].notna()].copy()

    if len(df_valid) == 0:
        st.error(
            "No valid results with actual prices. "
            "Check date ranges and ensure forecast dates have available data."
        )
        st.stop()

    # BRANCH: COMPARE BOTH DISTRIBUTIONS

    if distribution == "Compare Both":
        st.success("✅ Validation Complete!")

        hit_rate_normal = (
            df_valid["Within Bounds Normal"].sum() / len(df_valid) * 100
        )
        hit_rate_student_t = (
            df_valid["Within Bounds Student-t"].sum() / len(df_valid) * 100
        )

        # --- Top-level comparison metrics ---
        st.subheader("📊 Model Comparison Results")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(
                "Normal Distribution Hit Rate",
                f"{hit_rate_normal:.1f}%",
                delta=f"{hit_rate_normal - 90:.1f}% vs expected",
            )
        with c2:
            st.metric(
                "Student-t Distribution Hit Rate",
                f"{hit_rate_student_t:.1f}%",
                delta=f"{hit_rate_student_t - 90:.1f}% vs expected",
            )
        with c3:
            winner = (
                "Student-t"
                if hit_rate_student_t > hit_rate_normal
                else "Normal"
            )
            diff = abs(hit_rate_student_t - hit_rate_normal)
            st.metric("Better Model", winner, delta=f"+{diff:.1f}% hit rate")

        # --- Calibration analysis ---
        if not df_valid.empty:
            st.write("---")
            st.header("🔬 Calibration Analysis")

            df_normal_cal = rename_for_calibration(df_valid, "Normal")
            df_student_cal = rename_for_calibration(df_valid, "Student-t")

            calib_normal = calculate_calibration_metrics(df_normal_cal)
            calib_student = calculate_calibration_metrics(df_student_cal)

            # Regime table (use whichever has regime data)
            calib_for_regime = (
                calib_student
                if "regime_analysis" in calib_student
                else calib_normal
            )

            st.subheader("📊 Performance by Volatility Regime")
            render_regime_table(calib_for_regime)

            # Pick better model for detailed analysis
            if hit_rate_student_t >= hit_rate_normal:
                calib_best = calib_student
                dir_accuracy = calculate_directional_accuracy(
                    df_student_cal, num_days=num_days
                )
                failure_patterns = identify_failure_patterns(df_student_cal)
            else:
                calib_best = calib_normal
                dir_accuracy = calculate_directional_accuracy(
                    df_normal_cal, num_days=num_days
                )
                failure_patterns = identify_failure_patterns(df_normal_cal)

            st.write("---")
            st.subheader("🎯 Directional Accuracy")
            render_directional_accuracy(dir_accuracy)

            st.write("---")
            st.subheader("⚠️ Failure Case Analysis")
            render_failure_analysis(failure_patterns, len(df_valid))

            if calib_best["worst_misses"]:
                st.write("---")
                st.subheader("📉 Worst Prediction Misses")
                render_worst_misses(calib_best["worst_misses"])

        # --- Interval width analysis ---
        df_valid["Interval Width Normal"] = (
            (df_valid["Upper Bound Normal"] - df_valid["Lower Bound Normal"])
            / df_valid["Starting Price"]
            * 100
        )
        df_valid["Interval Width Student-t"] = (
            (
                df_valid["Upper Bound Student-t"]
                - df_valid["Lower Bound Student-t"]
            )
            / df_valid["Starting Price"]
            * 100
        )
        avg_width_normal = df_valid["Interval Width Normal"].mean()
        avg_width_student_t = df_valid["Interval Width Student-t"].mean()
        width_difference = (
            (avg_width_student_t - avg_width_normal) / avg_width_normal * 100
        )

        st.write("---")
        st.subheader("📏 Interval Width Analysis")
        st.write(
            f"**Normal Distribution:** Average interval width: "
            f"{avg_width_normal:.1f}% of stock price"
        )
        st.write(
            f"**Student-t Distribution:** Average interval width: "
            f"{avg_width_student_t:.1f}% of stock price"
        )
        st.write(
            f"**Difference:** Student-t intervals are "
            f"{width_difference:.1f}% wider"
        )

        # Recommendation
        if hit_rate_student_t >= 90 and hit_rate_normal < 90:
            st.success(
                f"✅ **Recommendation:** Use Student-t distribution. "
                f"It achieves {hit_rate_student_t:.1f}% hit rate vs "
                f"Normal's {hit_rate_normal:.1f}%."
            )
        elif hit_rate_normal >= 90 and hit_rate_student_t < 90:
            st.info(
                f"ℹ️ **Recommendation:** Normal distribution is sufficient. "
                f"It achieves {hit_rate_normal:.1f}% hit rate with "
                f"narrower intervals."
            )
        elif hit_rate_student_t > hit_rate_normal:
            st.info(
                f"ℹ️ **Recommendation:** Student-t performs better "
                f"({hit_rate_student_t:.1f}% vs {hit_rate_normal:.1f}%), "
                f"but at the cost of {width_difference:.1f}% wider intervals."
            )
        else:
            st.info(
                "ℹ️ **Recommendation:** Normal distribution is adequate "
                "for this stock/timeframe."
            )

        # --- Detailed comparison table ---
        st.write("---")
        st.subheader("📊 Detailed Comparison Table")
        comparison_df = pd.DataFrame(
            {
                "Metric": [
                    "Hit Rate",
                    "Avg Interval Width",
                    "Hits",
                    "Misses",
                ],
                "Normal": [
                    f"{hit_rate_normal:.1f}%",
                    f"{avg_width_normal:.1f}%",
                    int(df_valid["Within Bounds Normal"].sum()),
                    int(
                        len(df_valid)
                        - df_valid["Within Bounds Normal"].sum()
                    ),
                ],
                "Student-t": [
                    f"{hit_rate_student_t:.1f}%",
                    f"{avg_width_student_t:.1f}%",
                    int(df_valid["Within Bounds Student-t"].sum()),
                    int(
                        len(df_valid)
                        - df_valid["Within Bounds Student-t"].sum()
                    ),
                ],
            }
        )
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        # --- Rolling coverage stability ---
        st.write("---")
        st.subheader("📊 Calibration Stability Over Time")

        rolling_normal = calculate_rolling_coverage(
            df_valid.assign(
                **{"Within Bounds": df_valid["Within Bounds Normal"]}
            )
        )
        rolling_student_t = calculate_rolling_coverage(
            df_valid.assign(
                **{"Within Bounds": df_valid["Within Bounds Student-t"]}
            )
        )

        if (
            rolling_normal["sufficient_data"]
            and rolling_student_t["sufficient_data"]
        ):
            rc1, rc2 = st.columns(2)
            with rc1:
                st.write("**Normal Distribution:**")
                
                st.metric(
                        "Mean Coverage",
                        f"{rolling_normal['mean_coverage']:.1f}%",
                )
                st.metric(
                        "Worst Period",
                        f"{rolling_normal['min_coverage']:.1f}%",
                        delta=f"{rolling_normal['min_coverage'] - 90:.1f}%",
                )
                st.metric(
                        "Stability (±σ)",
                        f"{rolling_normal['std_coverage']:.1f}%",
                )
                if rolling_normal["min_coverage"] < 75:
                    st.warning(
                        f"⚠️ Dropped to {rolling_normal['min_coverage']:.1f}% "
                        f"from {rolling_normal['worst_period']['start_date']} "
                        f"to {rolling_normal['worst_period']['end_date']}"
                    )
            with rc2:
                st.write("**Student-t Distribution:**")
                st.metric(
                    "Mean Coverage",
                    f"{rolling_student_t['mean_coverage']:.1f}%",
                )
                st.metric(
                    "Worst Period",
                    f"{rolling_student_t['min_coverage']:.1f}%",
                    delta=f"{rolling_student_t['min_coverage'] - 90:.1f}%",
                )
                st.metric(
                    "Stability (±σ)",
                    f"{rolling_student_t['std_coverage']:.1f}%",
                )
                if rolling_student_t["min_coverage"] < 75:
                    st.warning(
                        f"⚠️ Dropped to "
                        f"{rolling_student_t['min_coverage']:.1f}% from "
                        f"{rolling_student_t['worst_period']['start_date']} "
                        f"to {rolling_student_t['worst_period']['end_date']}"
                    )
            st.caption(
                f"*Rolling window size: {rolling_normal['window_size']} tests*"
            )
        else:
            st.info(
                "Not enough data for rolling coverage analysis "
                "(need at least 5 backtests)"
            )

        # --- Comparison chart ---
        st.write("---")
        st.subheader("📈 Validation Over Time - Both Models")

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 12), facecolor="#0e1117"
        )
        dates = pd.to_datetime(df_valid["Test Date"])

        plot_validation_subplot(
            ax1, dates, df_valid,
            "Lower Bound Normal", "Upper Bound Normal",
            "Within Bounds Normal", "#4A90E2",
            f"Normal Distribution - Hit Rate: {hit_rate_normal:.1f}%",
        )
        ax1.set_ylabel("Price ($)", fontsize=12, fontweight="bold", color="white")

        plot_validation_subplot(
            ax2, dates, df_valid,
            "Lower Bound Student-t", "Upper Bound Student-t",
            "Within Bounds Student-t", "#E74C3C",
            f"Student-t Distribution (df={df_param}) - "
            f"Hit Rate: {hit_rate_student_t:.1f}%",
        )
        ax2.set_xlabel("Test Date", fontsize=12, fontweight="bold", color="white")
        ax2.set_ylabel("Price ($)", fontsize=12, fontweight="bold", color="white")

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    # BRANCH: SINGLE DISTRIBUTION

    else:
        st.success("✅ Backtest Complete!")

        st.info(
            """
            📊 **What is Hit Rate?**

            The model generates a 90% confidence interval (5th to 95th percentile).
            If the model is well-calibrated, actual prices should fall within
            this range ~90% of the time.

            - **Hit Rate > 85%**: Excellent calibration ✅
            - **Hit Rate 75-85%**: Good calibration ✓
            - **Hit Rate < 75%**: Model may need adjustment ⚠️
            """
        )

        hit_rate = df_valid["Within Bounds"].sum() / len(df_valid) * 100
        num_hits = df_valid["Within Bounds"].sum()
        num_misses = len(df_valid) - num_hits
        above_upper = (
            df_valid["Actual Price"] > df_valid["Upper Bound"]
        ).sum()
        below_lower = (
            df_valid["Actual Price"] < df_valid["Lower Bound"]
        ).sum()

        # --- Top metrics ---
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(
                "Hit Rate",
                f"{hit_rate:.1f}%",
                help="Percentage of times actual price fell within "
                "90% confidence interval",
            )
        with c2:
            st.metric(
                "Successful Tests",
                int(num_hits),
                delta=f"out of {len(df_valid)} valid",
            )
        with c3:
            st.metric(
                "Total Tests",
                len(df_results),
                delta=f"{len(df_results) - len(df_valid)} skipped",
            )

        # --- Diagnostics ---
        avg_interval_width = (
            (df_valid["Upper Bound"] - df_valid["Lower Bound"])
            / df_valid["Starting Price"]
            * 100
        ).mean()

        n = len(df_valid)
        margin_of_error = (
            1.96 * np.sqrt(0.9 * 0.1 / n) * 100 if n > 0 else 0
        )
        p_value = (
            binomtest(int(num_hits), n, 0.9, alternative="two-sided").pvalue
            if n > 0
            else 1.0
        )

        st.write("---")
        st.subheader("📊 Diagnostic Information")
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.metric(
                "Above Upper Bound",
                int(above_upper),
                help="Times actual price exceeded 95th percentile",
            )
        with d2:
            st.metric(
                "Below Lower Bound",
                int(below_lower),
                help="Times actual price fell below 5th percentile",
            )
        with d3:
            st.metric(
                "Avg Interval Width",
                f"{avg_interval_width:.1f}%",
                help="Average width of 90% CI as % of starting price",
            )
        with d4:
            st.metric(
                "Expected Range",
                f"90.0% ± {margin_of_error:.1f}%",
                help=f"Expected hit rate for {n} samples (95% confidence)",
            )

        # --- Statistical analysis ---
        st.write("---")
        st.subheader("📈 Statistical Analysis")

        sa, sb = st.columns(2)
        with sa:
            st.write("**Statistical Significance:**")
            st.write(f"- P-value (binomial test): {p_value:.4f}")
            if p_value < 0.05:
                st.warning(
                    "⚠️ Statistically significant deviation from 90%"
                )
            else:
                st.success(
                    "✅ Not statistically significant (within expected range)"
                )
            st.write(f"- Margin of error (95% CI): ±{margin_of_error:.1f}%")
            st.write(f"- Your result: {hit_rate:.1f}%")
        with sb:
            st.write("**Model Performance:**")
            if hit_rate >= 85:
                st.success(
                    f"✅ Excellent! Hit rate of {hit_rate:.1f}% indicates "
                    "well-calibrated model."
                )
            elif hit_rate >= 75:
                st.info(
                    f"✓ Good. Hit rate of {hit_rate:.1f}% is close to "
                    "expected 90%."
                )
            else:
                st.warning(
                    f"⚠️ Hit rate of {hit_rate:.1f}% is below expected. "
                    "Model may need adjustment."
                )
        

        # --- Advanced calibration expander ---
        st.write("---")
        st.subheader("📊 Calibration Report")

        with st.expander("🔬 Advanced Calibration Analysis", expanded=False):
            calibration = calculate_calibration_metrics(df_valid)

            st.write("**Volatility Regime Analysis:**")
            if calibration["regime_analysis"]:
                regime_data = []
                for regime, s in calibration["regime_analysis"].items():
                    regime_data.append(
                        {
                            "Regime": regime,
                            "Hit Rate": f"{s['hit_rate']:.1f}%",
                            "Tests": s["count"],
                            "Avg Miss Magnitude": f"{s['avg_miss_magnitude']:.1f}%",
                        }
                    )
                st.dataframe(
                    pd.DataFrame(regime_data),
                    use_container_width=True,
                    hide_index=True,
                )
                for regime, s in calibration["regime_analysis"].items():
                    if s["hit_rate"] < 80:
                        st.warning(
                            f"⚠️ **{regime} Volatility:** Only "
                            f"{s['hit_rate']:.1f}% hit rate. Model struggles "
                            f"in {regime.lower()} volatility conditions."
                        )
                    elif s["hit_rate"] > 95:
                        st.info(
                            f"ℹ️ **{regime} Volatility:** "
                            f"{s['hit_rate']:.1f}% hit rate may indicate "
                            "overly conservative intervals."
                        )
            else:
                st.info(
                    "Enable volatility tracking to see regime-based analysis"
                )

            st.write("---")
            st.write("**Directional Accuracy:**")
            directional = calculate_directional_accuracy(df_valid, num_days)
            dc1, dc2, dc3 = st.columns(3)
            with dc1:
                st.metric(
                    "Overall Direction Accuracy",
                    f"{directional['overall_accuracy']*100:.1f}%",
                )
            with dc2:
                st.metric(
                    "Upward Move Accuracy",
                    f"{directional['upward_accuracy']*100:.1f}%",
                )
            with dc3:
                st.metric(
                    "Downward Move Accuracy",
                    f"{directional['downward_accuracy']*100:.1f}%",
                )
            if directional["overall_accuracy"] < 0.5:
                st.warning(
                    "⚠️ Directional accuracy below 50% suggests drift "
                    "parameter may need adjustment"
                )

            st.write("---")
            st.write("**Failure Pattern Analysis:**")
            failures = identify_failure_patterns(df_valid, threshold=0.15)
            fc1, fc2 = st.columns(2)
            with fc1:
                st.metric(
                    "Major Failures (>15% miss)",
                    failures["major_failures"],
                )
                st.metric(
                    "Failure Rate", f"{failures['failure_rate']:.1f}%"
                )
            with fc2:
                st.metric("Total Misses", failures["total_misses"])
                st.metric(
                    "Avg Miss Magnitude",
                    f"{failures['avg_miss_magnitude']*100:.1f}%",
                )
            if failures["patterns"]:
                st.write("**Detected Patterns:**")
                for p in failures["patterns"]:
                    st.write(f"- {p}")
            
            st.write("---")
            st.write("**Calibration Stability Over Time:**")
            rolling = calculate_rolling_coverage(df_valid)

            if rolling["sufficient_data"]:
                sc1, sc2, sc3 = st.columns(3)
                with sc1:
                    st.metric(
                        "Mean Coverage",
                        f"{rolling['mean_coverage']:.1f}%",
                    )
                with sc2:
                    st.metric(
                        "Worst Period",
                        f"{rolling['min_coverage']:.1f}%",
                        delta=f"{rolling['min_coverage'] - 90:.1f}%",
                    )
                with sc3:
                    st.metric(
                        "Stability (±σ)",
                        f"{rolling['std_coverage']:.1f}%",
                        help="Lower is more stable",
                    )

                if rolling["min_coverage"] < 75:
                    st.warning(
                        f"⚠️ Coverage dropped to "
                        f"{rolling['min_coverage']:.1f}% during "
                        f"{rolling['worst_period']['start_date']} to "
                        f"{rolling['worst_period']['end_date']}"
                    )

                st.caption(
                    f"*Rolling window size: {rolling['window_size']} tests*"
                )
            else:
                st.info(
                    "Not enough data for rolling coverage analysis "
                    "(need at least 5 tests)"
                )

            st.write("---")
            st.write("**Worst Misses:**")
            if calibration["worst_misses"]:
                worst_df = pd.DataFrame(calibration["worst_misses"])
                if "Test Date" in worst_df.columns:
                    worst_df = worst_df[
                        [
                            "Test Date",
                            "Actual Price",
                            "Lower Bound",
                            "Upper Bound",
                            "Miss Magnitude",
                            "Volatility",
                        ]
                    ]
                worst_df["Actual Price"] = worst_df["Actual Price"].apply(
                    lambda x: f"${x:.2f}"
                )
                worst_df["Lower Bound"] = worst_df["Lower Bound"].apply(
                    lambda x: f"${x:.2f}"
                )
                worst_df["Upper Bound"] = worst_df["Upper Bound"].apply(
                    lambda x: f"${x:.2f}"
                )
                worst_df["Miss Magnitude"] = worst_df[
                    "Miss Magnitude"
                ].apply(lambda x: f"{x:.1f}%")
                worst_df["Volatility"] = worst_df["Volatility"].apply(
                    lambda x: f"{x:.1f}%"
                )
                st.dataframe(
                    worst_df.head(5),
                    use_container_width=True,
                    hide_index=True,
                )

        # --- Sample misses ---
        if num_misses > 0:
            st.write("---")
            st.subheader("🔍 Sample Misses")
            misses_df = df_valid[~df_valid["Within Bounds"]].head(3)
            for _, row in misses_df.iterrows():
                with st.expander(f"Miss on {row['Test Date']}"):
                    st.write(
                        f"**Starting Price:** ${row['Starting Price']:.2f}"
                    )
                    st.write(f"**Actual Price:** ${row['Actual Price']:.2f}")
                    st.write(
                        f"**Bounds:** [${row['Lower Bound']:.2f}, "
                        f"${row['Upper Bound']:.2f}]"
                    )
                    if row["Actual Price"] < row["Lower Bound"]:
                        st.write(
                            f"**Result:** Below lower bound by "
                            f"${row['Lower Bound'] - row['Actual Price']:.2f}"
                        )
                    else:
                        st.write(
                            f"**Result:** Above upper bound by "
                            f"${row['Actual Price'] - row['Upper Bound']:.2f}"
                        )

        # --- Validation chart ---
        st.write("---")
        st.subheader("📈 Validation Over Time")

        fig, ax = plt.subplots(figsize=(14, 7), facecolor="#0e1117")
        dates = pd.to_datetime(df_valid["Test Date"])

        plot_validation_subplot(
            ax, dates, df_valid,
            "Lower Bound", "Upper Bound", "Within Bounds", "#4A90E2",
            f"Backtest Validation: {ticker} - "
            f"{timeframe.replace('_', ' ').title()}",
        )
        ax.set_xlabel(
            "Test Date", fontsize=13, fontweight="bold", color="white"
        )
        ax.set_ylabel(
            "Price ($)", fontsize=13, fontweight="bold", color="white"
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.xticks(rotation=45, ha="right", color="white")
        plt.yticks(color="white")
        plt.tight_layout(pad=1.0)
        st.pyplot(fig, use_container_width=True)

        # --- Detailed results table ---
        st.write("---")
        st.subheader("📋 Detailed Results")

        df_display = df_valid.copy()
        df_display["Actual Price"] = df_display["Actual Price"].apply(
            lambda x: f"${x:.2f}"
        )
        df_display["Lower Bound"] = df_display["Lower Bound"].apply(
            lambda x: f"${x:.2f}"
        )
        df_display["Upper Bound"] = df_display["Upper Bound"].apply(
            lambda x: f"${x:.2f}"
        )
        df_display["Within Bounds"] = df_display["Within Bounds"].apply(
            lambda x: "✓" if x else "✗"
        )
        df_display["Volatility"] = df_display["Volatility"].apply(
            lambda x: f"{x:.1f}%"
        )

        display_cols = [
            "Test Date",
            "Forecast Date",
            "Starting Price",
            "Actual Price",
            "Lower Bound",
            "Upper Bound",
            "Within Bounds",
            "Volatility",
        ]
        st.dataframe(df_display[display_cols], use_container_width=True)