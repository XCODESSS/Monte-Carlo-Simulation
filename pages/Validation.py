import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy.stats import binomtest
from Monte_Carlo import *
import os

@st.cache_data(ttl=3600, show_spinner=False)
def cached_download_stock_data(ticker, start_date, end_date):
    """Cached wrapper for download_stock_data"""
    return download_stock_data(ticker, start_date, end_date)

# Configure matplotlib/seaborn for dark theme
plt.rcParams['figure.facecolor'] = '#0e1117'
plt.rcParams['axes.facecolor'] = '#0e1117'
plt.rcParams['savefig.facecolor'] = '#0e1117'
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['axes.edgecolor'] = '#2d2d2d'
plt.rcParams['grid.color'] = '#2d2d2d'
sns.set_style("darkgrid", {'axes.facecolor': '#0e1117', 'figure.facecolor': '#0e1117'})

st.title("📊 Model Validation Dashboard")
st.write("Backtest the Monte Carlo model on historical data to verify accuracy")

# User inputs
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

    if ticker.endswith('.NS') and not alpha_vantage_key:
        alpha_vantage_key = st.text_input("Alpha Vantage API Key", type="password", help="Required for Indian stocks (.NS)")
        if not alpha_vantage_key:
            st.warning("⚠️ Alpha Vantage API key required for Indian stocks.")
    timeframe = st.selectbox("Forecast Timeframe", list(TIMEFRAMES.keys()))
    num_backtests = st.slider("Number of Backtests", min_value=10, max_value=100, value=30, step=5)

    # Distribution Selection
    st.write("---")
    distribution = st.selectbox(
        "Distribution Model",
        ["Normal", 'Student-t (Fat tails)', 'Compare Both'],
        help = "Choose which distribution to validate"
    )

    if distribution == 'Student-t (Fat tails)' or distribution == "Compare Both":
        df = st.slider(
            "Degree of Freedom ",
            min_value = 3,
            max_value = 30,
            value = 7,
            help = "Lower values = fatter tails(more extreme events)"

        )
    else:
        df = None



with col2:
    backtest_end = st.date_input(
        "Backtest End Date",
        value=pd.Timestamp('2023-01-01'),
        help="Latest date to run backtest from. For COVID-19 testing, use 2020-02-15"
    )
    historical_start = st.date_input(
        "Historical Data Start",
        value=pd.Timestamp('2015-01-01'),
        help="How far back to use for parameter estimation"
    )

    st.write("---")
    use_rolling_window = st.checkbox(
        "Use Rolling Window",
        value=True,
        help="Use only recent data for parameter estimation (better for short-term forecasts)"
    )

    if use_rolling_window:
        lookback_days = st.number_input(
            "Lookback Period (Days)",
            min_value=30,
            max_value=252*5,
            value=252,
            step=10,
            help="Number of trading days to use for calculating volatility and drift"
        )
    else:
        lookback_days = None

# Validate date inputs
if historical_start >= backtest_end:
    st.error("❌ Historical Data Start must be before Backtest End Date")
    st.stop()

if st.button("Run Backtest", type="primary"):
    num_days = TIMEFRAMES[timeframe]

    # Use datetime objects from date_input widgets
    end_date = pd.Timestamp(backtest_end)
    start_date_str = historical_start.strftime('%Y-%m-%d')
    end_date_str = backtest_end.strftime('%Y-%m-%d')

    # Generate test dates going BACKWARDS from end_date (so we test predictions made before the end date)
    # This ensures we're testing predictions made in the past, not the future
    test_dates = pd.date_range(end=end_date, periods=num_backtests, freq='ME')
    test_dates = test_dates.sort_values()  # Ensure chronological order

    # Validation check
    try:
        test_data = cached_download_stock_data(ticker, start_date_str, end_date_str)
        if test_data.empty:
            st.error(f"❌ No data found for ticker '{ticker}'. Please check the symbol.")
            st.stop()
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("💡 Tip: Make sure you're using a valid stock ticker")
        st.stop()

    results = []
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, test_date in enumerate(test_dates):
        status_text.text(f"Running backtest {i+1}/{num_backtests}...")
        progress_bar.progress((i + 1) / num_backtests)

        try:
            # Download data UP TO test_date only
            data = download_stock_data(ticker, start_date_str, test_date.strftime('%Y-%m-%d'), api_key=alpha_vantage_key)

             # Skip if insufficient data (need at least 30 days for meaningful statistics)
            if len(data) < 30:
                continue

            # Apply Rolling Window logic
            if use_rolling_window and lookback_days:
                if len(data) > lookback_days:
                    stats_data = data.tail(lookback_days)
                else:
                    stats_data = data
            else:
                stats_data = data

            # Calculate statistics
            stats = calculate_statistics(stats_data)
            
           # Run simulation(s) based on distribution choice
            if distribution == "Normal":
                simulations = run_monte_carlo(
                    stats['starting_price'],
                    stats['mu'],
                    stats['sigma'],
                    num_days,
                    1000
                )
                metrics = calculate_metrics(
                    simulations,
                    stats['starting_price'],
                    stats['mu'],
                    stats['sigma'],
                    num_days
                )
                lower_bound = metrics['lower_bound']
                upper_bound = metrics['upper_bound']

            elif distribution == "Student-t (Fat tails)":
                simulations = run_monte_carlo_student_t(
                    stats['starting_price'],
                    stats['mu'],
                    stats['sigma'],
                    num_days,
                    1000,
                    df=df
                )
                metrics = calculate_metrics(
                    simulations,
                    stats['starting_price'],
                    stats['mu'],
                    stats['sigma'],
                    num_days
                )
                lower_bound = metrics['lower_bound']
                upper_bound = metrics['upper_bound']

            else:  # Compare Both
                # Run both models
                simulations_normal = run_monte_carlo(
                    stats['starting_price'],
                    stats['mu'],
                    stats['sigma'],
                    num_days,
                    1000
                )
                simulations_student_t = run_monte_carlo_student_t(
                    stats['starting_price'],
                    stats['mu'],
                    stats['sigma'],
                    num_days,
                    1000,
                    df=df
                )

                metrics_normal = calculate_metrics(
                    simulations_normal,
                    stats['starting_price'],
                    stats['mu'],
                    stats['sigma'],
                    num_days
                )
                metrics_student_t = calculate_metrics(
                    simulations_student_t,
                    stats['starting_price'],
                    stats['mu'],
                    stats['sigma'],
                    num_days
                )

                # For comparison mode, we'll track both
                lower_bound_normal = metrics_normal['lower_bound']
                upper_bound_normal = metrics_normal['upper_bound']
                lower_bound_student_t = metrics_student_t['lower_bound']
                upper_bound_student_t = metrics_student_t['upper_bound']

            # Get actual price at exactly num_days trading days after test_date
            # We need to download data starting from test_date and find the price num_days trading days later
            # Add buffer to account for weekends/holidays (approximately 1.4x for trading days)
            calendar_days_buffer = int(num_days * 1.4) + 20  # Extra buffer to ensure we have enough data
            forecast_date = test_date + pd.Timedelta(days=calendar_days_buffer)

            # Download data from test_date onwards to get the future price
            actual_data = download_stock_data(
                ticker,
                test_date.strftime('%Y-%m-%d'),
                forecast_date.strftime('%Y-%m-%d'),
                api_key=alpha_vantage_key
            )

            # IMPORTANT: Get the price at exactly num_days trading days after test_date
            # yfinance returns data starting from test_date (if it's a trading day) or the next trading day
            # We need to find the price that is exactly num_days trading days in the future

            if len(actual_data) > num_days:
                # The data starts at test_date (or next trading day)
                # We want the price at num_days trading days later
                # If test_date is included: iloc[0] = test_date (day 0), iloc[num_days] = num_days later
                # If test_date is not a trading day: iloc[0] = next trading day (day 1), iloc[num_days-1] = num_days later

                # Check if test_date is in the data (it's a trading day)
                test_date_in_data = test_date in actual_data.index or test_date.date() in [d.date() for d in actual_data.index]

                if test_date_in_data or actual_data.index[0].date() == test_date.date():
                    # test_date is included, so iloc[num_days] gives us num_days trading days later
                    actual_price = float(actual_data['Close'].iloc[num_days])
                    actual_forecast_date = actual_data.index[num_days]
                else:
                    # test_date is not a trading day, so iloc[0] is already 1 trading day later
                    # We need iloc[num_days - 1] to get num_days trading days after test_date
                    actual_price = float(actual_data['Close'].iloc[num_days - 1])
                    actual_forecast_date = actual_data.index[num_days - 1]

                if distribution == "Compare Both":
                    actual_within_bounds = None # Not used in compare mode
                else:
                    actual_within_bounds = lower_bound <= actual_price <= upper_bound
            elif len(actual_data) >= num_days:
                # Exactly num_days or slightly more - use the last valid index
                idx_to_use = min(num_days, len(actual_data) - 1)
                actual_price = float(actual_data['Close'].iloc[idx_to_use])
                actual_forecast_date = actual_data.index[idx_to_use]
                if distribution == "Compare Both":
                    actual_within_bounds = None
                else:
                    actual_within_bounds = lower_bound <= actual_price <= upper_bound
            else:
                # Not enough data - skip this test
                actual_price = None
                actual_within_bounds = False
                actual_forecast_date = None

            # Store results
            if distribution == "Compare Both":
                # Check if actual price is within bounds for BOTH models
                if actual_price is not None:
                    within_bounds_normal = (lower_bound_normal <= actual_price <= upper_bound_normal)
                    within_bounds_student_t = (lower_bound_student_t <= actual_price <= upper_bound_student_t)
                else:
                    within_bounds_normal = False
                    within_bounds_student_t = False

                results.append({
                    'Test Date': test_date.strftime('%Y-%m-%d'),
                    'Forecast Date': actual_forecast_date.strftime('%Y-%m-%d') if actual_forecast_date else 'N/A',
                    'Starting Price': stats['starting_price'],
                    'Actual Price': actual_price,
                    'Lower Bound Normal': lower_bound_normal,
                    'Upper Bound Normal': upper_bound_normal,
                    'Lower Bound Student-t': lower_bound_student_t,
                    'Upper Bound Student-t': upper_bound_student_t,
                    'Within Bounds Normal': within_bounds_normal,
                    'Within Bounds Student-t': within_bounds_student_t,
                    'Volatility': stats['sigma'] * np.sqrt(252) * 100,
                    'Distribution': 'Both'
                })
            else:
                # Single model (Normal or Student-t)
                if actual_price is not None:
                    within_bounds = (lower_bound <= actual_price <= upper_bound)
                else:
                    within_bounds = False

                results.append({
                    'Test Date': test_date.strftime('%Y-%m-%d'),
                    'Forecast Date': actual_forecast_date.strftime('%Y-%m-%d') if actual_forecast_date else 'N/A',
                    'Starting Price': stats['starting_price'],
                    'Actual Price': actual_price,
                    'Lower Bound': lower_bound,
                    'Upper Bound': upper_bound,
                    'Within Bounds': within_bounds,
                    'Volatility': stats['sigma'] * np.sqrt(252) * 100,
                    'Distribution': distribution
                })

            
        except Exception as e:
            st.error(f"Error at test {i+1} ({test_date}): {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            continue

    progress_bar.empty()
    status_text.empty()

    if results:
        # Convert results to DataFrame
        df_results = pd.DataFrame(results)

        # Filter out None values for accurate hit rate calculation
        df_valid = df_results[df_results['Actual Price'].notna()].copy()

        if len(df_valid) == 0:
            st.error("No valid results with actual prices. Check date ranges and ensure forecast dates have available data.")
            st.stop()
            
        # Use filtered dataframe for display logic
        df = df_valid

        if distribution == "Compare Both":
            # Comparison mode display
            st.success("✅ Validation Complete!")

            # Calculate hit rates for both models
            hit_rate_normal = (df['Within Bounds Normal'].sum() / len(df)) * 100
            hit_rate_student_t = (df['Within Bounds Student-t'].sum() / len(df)) * 100

            st.subheader("📊 Model Comparison Results")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Normal Distribution Hit Rate",
                    f"{hit_rate_normal:.1f}%",
                    delta=f"{hit_rate_normal - 90:.1f}% vs expected"
                )
            with col2:
                st.metric(
                    "Student-t Distribution Hit Rate",
                    f"{hit_rate_student_t:.1f}%",
                    delta=f"{hit_rate_student_t - 90:.1f}% vs expected"
                )
            with col3:
                winner = "Student-t" if hit_rate_student_t > hit_rate_normal else "Normal"
                diff = abs(hit_rate_student_t - hit_rate_normal)
                st.metric(
                    "Better Model",
                    winner,
                    delta=f"+{diff:.1f}% hit rate"
                )

            # Calculate interval widths
            df['Interval Width Normal'] = (df['Upper Bound Normal'] - df['Lower Bound Normal']) / df['Starting Price'] * 100
            df['Interval Width Student-t'] = (df['Upper Bound Student-t'] - df['Lower Bound Student-t']) / df['Starting Price'] * 100

            avg_width_normal = df['Interval Width Normal'].mean()
            avg_width_student_t = df['Interval Width Student-t'].mean()
            width_difference = ((avg_width_student_t - avg_width_normal) / avg_width_normal) * 100

            st.write("---")
            st.subheader("📏 Interval Width Analysis")
            st.write(f"**Normal Distribution:** Average interval width: {avg_width_normal:.1f}% of stock price")
            st.write(f"**Student-t Distribution:** Average interval width: {avg_width_student_t:.1f}% of stock price")
            st.write(f"**Difference:** Student-t intervals are {width_difference:.1f}% wider")

            if hit_rate_student_t >= 90 and hit_rate_normal < 90:
                st.success(f"✅ **Recommendation:** Use Student-t distribution. It achieves {hit_rate_student_t:.1f}% hit rate vs Normal's {hit_rate_normal:.1f}%.")
            elif hit_rate_normal >= 90 and hit_rate_student_t < 90:
                st.info(f"ℹ️ **Recommendation:** Normal distribution is sufficient. It achieves {hit_rate_normal:.1f}% hit rate with narrower intervals.")
            elif hit_rate_student_t > hit_rate_normal:
                st.info(f"ℹ️ **Recommendation:** Student-t performs better ({hit_rate_student_t:.1f}% vs {hit_rate_normal:.1f}%), but at the cost of {width_difference:.1f}% wider intervals.")
            else:
                st.info(f"ℹ️ **Recommendation:** Normal distribution is adequate for this stock/timeframe.")

        else:
            # Calculate hit rate only on valid results
            hit_rate = (df_valid['Within Bounds'].sum() / len(df_valid)) * 100
            
            # Calculate additional diagnostics
            num_hits = df_valid['Within Bounds'].sum()
            num_misses = len(df_valid) - num_hits
            above_upper = (df_valid['Actual Price'] > df_valid['Upper Bound']).sum()
            below_lower = (df_valid['Actual Price'] < df_valid['Lower Bound']).sum()

            # Calculate average interval width as percentage of starting price
            avg_interval_width = ((df_valid['Upper Bound'] - df_valid['Lower Bound']) / df_valid['Starting Price'] * 100).mean()

            # Calculate how close actual prices are to bounds (to detect if intervals are too wide)
            df_valid_copy = df_valid.copy()
            df_valid_copy['Distance to Lower'] = (df_valid_copy['Actual Price'] - df_valid_copy['Lower Bound']) / df_valid_copy['Starting Price'] * 100
            df_valid_copy['Distance to Upper'] = (df_valid_copy['Upper Bound'] - df_valid_copy['Actual Price']) / df_valid_copy['Starting Price'] * 100
            df_valid_copy['Distance to Center'] = abs(df_valid_copy['Actual Price'] - (df_valid_copy['Lower Bound'] + df_valid_copy['Upper Bound']) / 2) / df_valid_copy['Starting Price'] * 100

            avg_distance_to_lower = df_valid_copy['Distance to Lower'].mean()
            avg_distance_to_upper = df_valid_copy['Distance to Upper'].mean()
            avg_distance_to_center = df_valid_copy['Distance to Center'].mean()

            # Check if intervals are suspiciously wide (actual prices clustering near center)
            prices_near_center = (df_valid_copy['Distance to Center'] < avg_interval_width * 0.1).sum()
            center_clustering_ratio = prices_near_center / len(df_valid_copy) if len(df_valid_copy) > 0 else 0

            # Statistical note: For a 90% CI, expected hit rate is 90% ± margin of error
            # With n samples, margin of error ≈ 1.96 * sqrt(0.9 * 0.1 / n) * 100
            n = len(df_valid)
            expected_hit_rate = 90.0
            margin_of_error = 1.96 * np.sqrt(0.9 * 0.1 / n) * 100 if n > 0 else 0
            is_within_expected = abs(hit_rate - expected_hit_rate) <= margin_of_error

            # Calculate p-value for this hit rate (binomial test)
            # H0: true hit rate = 90%, H1: true hit rate != 90%
            from scipy.stats import binomtest
            p_value = binomtest(num_hits, n, 0.9, alternative='two-sided').pvalue if n > 0 else 1.0

            # Display summary
            st.success("Backtest Complete!")
            # After st.success("Backtest Complete!")
            st.info("""
            📊 **What is Hit Rate?**
    
            The model generates a 90% confidence interval (5th to 95th percentile).
            If the model is well-calibrated, actual prices should fall within this range ~90% of the time.
    
            - **Hit Rate > 85%**: Excellent calibration ✅
            - **Hit Rate 75-85%**: Good calibration ✓
            - **Hit Rate < 75%**: Model may need adjustment ⚠️
            """)
            # Display key metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Hit Rate", f"{hit_rate:.1f}%",
                        help="Percentage of times actual price fell within 90% confidence interval")
            with col2:
                st.metric("Successful Tests", df_valid['Within Bounds'].sum(),
                        delta=f"out of {len(df_valid)} valid")
            with col3:
                st.metric("Total Tests", len(df_results),
                        delta=f"{len(df_results) - len(df_valid)} skipped")

            # Additional Diagnostics
            st.write("---")
            st.subheader("📊 Diagnostic Information")

            diag_col1, diag_col2, diag_col3, diag_col4 = st.columns(4)
            with diag_col1:
                st.metric("Above Upper Bound", above_upper,
                        help="Number of times actual price exceeded 95th percentile")
            with diag_col2:
                st.metric("Below Lower Bound", below_lower,
                        help="Number of times actual price fell below 5th percentile")
            with diag_col3:
                st.metric("Avg Interval Width", f"{avg_interval_width:.1f}%",
                        help="Average width of 90% confidence interval as % of starting price")
            with diag_col4:
                st.metric("Expected Range", f"90.0% ± {margin_of_error:.1f}%",
                        help=f"Expected hit rate for {n} samples (95% confidence)")

            # Detailed analysis
            st.write("---")
            st.subheader("🔍 Detailed Analysis")

            analysis_col1, analysis_col2 = st.columns(2)
            with analysis_col1:
                st.write("**Price Distribution Relative to Bounds:**")
                st.write(f"- Average distance to lower bound: {avg_distance_to_lower:.2f}% of starting price")
                st.write(f"- Average distance to upper bound: {avg_distance_to_upper:.2f}% of starting price")
                st.write(f"- Average distance to center: {avg_distance_to_center:.2f}% of starting price")
                st.write(f"- Prices near center (<10% of interval width): {prices_near_center} ({center_clustering_ratio*100:.1f}%)")

            with analysis_col2:
                st.write("**Statistical Significance:**")
                st.write(f"- P-value (binomial test): {p_value:.4f}")
                if p_value < 0.05:
                    st.warning("⚠️ **Statistically Significant Deviation**: The hit rate significantly differs from 90% (p < 0.05). This suggests the confidence intervals may need adjustment.")
                else:
                    st.success("✅ **Not Statistically Significant**: The deviation from 90% could be due to random variation.")
                st.write(f"- Margin of error (95% CI): ±{margin_of_error:.1f}%")
                st.write(f"- Your result: {hit_rate:.1f}%")

            # Statistical interpretation
            st.write("---")
            if hit_rate > 95 and p_value < 0.05:
                st.warning(f"⚠️ **Unusually High Hit Rate**: {hit_rate:.1f}% is significantly higher than the expected 90% (p = {p_value:.4f}). This suggests:")
                st.write("""
                1. **Confidence intervals may be too wide** - The model is being overly conservative
                2. **Possible date indexing issue** - Verify that actual prices are being compared at the correct forecast dates
                3. **Model may be well-calibrated** - But intervals are wider than necessary for risk management

                **Recommendation**: Check a few individual test cases to verify the actual prices are being compared correctly.
                """)
            elif is_within_expected:
                st.success(f"✅ **Statistically Normal**: Hit rate of {hit_rate:.1f}% is within the expected range of {expected_hit_rate:.1f}% ± {margin_of_error:.1f}% for {n} samples.")
            elif hit_rate > expected_hit_rate + margin_of_error:
                st.info(f"ℹ️ **Slightly Conservative**: Hit rate of {hit_rate:.1f}% is higher than expected. This suggests the confidence intervals may be slightly wider than needed, but this is acceptable for risk management.")
            else:
                st.warning(f"⚠️ **Below Expected**: Hit rate of {hit_rate:.1f}% is below the expected range. The model may need adjustment or the period tested may have higher volatility than historical data suggests.")

            # Interpretation
            st.write("---")
            st.subheader("Interpretation")
            if hit_rate >= 85:
                st.success(f"✅ Excellent! The model's 90% confidence intervals captured actual prices {hit_rate:.1f}% of the time. This suggests the model is well-calibrated.")
            elif hit_rate >= 75:
                st.info(f"✓ Good. The model captured actual prices {hit_rate:.1f}% of the time. Close to the expected 90%.")
            else:
                st.warning(f"⚠️ The model only captured actual prices {hit_rate:.1f}% of the time. This is below the expected 90%, suggesting the model may need adjustment.")

            # Show example cases for verification (especially misses)
            st.write("---")
            st.subheader("🔍 Sample Cases for Verification")

            if num_misses > 0:
                st.write("**Cases where actual price was OUTSIDE bounds:**")
                misses_df = df_valid[~df_valid['Within Bounds']].copy()
                misses_display = misses_df[['Test Date', 'Forecast Date', 'Starting Price', 'Actual Price',
                                        'Lower Bound', 'Upper Bound', 'Volatility']].head(5)
                for idx, row in misses_display.iterrows():
                    st.write(f"**Test Date: {row['Test Date']}** → Forecast Date: {row['Forecast Date']}")
                    st.write(f"  - Starting Price: ${row['Starting Price']:.2f}")
                    st.write(f"  - Actual Price: ${row['Actual Price']:.2f}")
                    st.write(f"  - Bounds: [${row['Lower Bound']:.2f}, ${row['Upper Bound']:.2f}]")
                    if row['Actual Price'] < row['Lower Bound']:
                        st.write(f"  - ❌ Below lower bound by ${row['Lower Bound'] - row['Actual Price']:.2f} ({(row['Lower Bound'] - row['Actual Price'])/row['Starting Price']*100:.1f}%)")
                    else:
                        st.write(f"  - ❌ Above upper bound by ${row['Actual Price'] - row['Upper Bound']:.2f} ({(row['Actual Price'] - row['Upper Bound'])/row['Starting Price']*100:.1f}%)")
                    st.write("")
            else:
                st.write("**All cases were within bounds.** Showing a few random examples:")
                sample_df = df_valid.sample(min(3, len(df_valid))).copy()
                for idx, row in sample_df.iterrows():
                    st.write(f"**Test Date: {row['Test Date']}** → Forecast Date: {row['Forecast Date']}")
                    st.write(f"  - Starting Price: ${row['Starting Price']:.2f}")
                    st.write(f"  - Actual Price: ${row['Actual Price']:.2f}")
                    st.write(f"  - Bounds: [${row['Lower Bound']:.2f}, ${row['Upper Bound']:.2f}]")
                    st.write(f"  - ✅ Within bounds (distance to center: {abs(row['Actual Price'] - (row['Lower Bound'] + row['Upper Bound'])/2)/row['Starting Price']*100:.1f}%)")
                    st.write("")

            # Display detailed results
            st.write("---")
            st.subheader("Detailed Results")

            # Format the dataframe for display (show only valid results)
            df_display = df_valid.copy()
            df_display['Actual Price'] = df_display['Actual Price'].apply(lambda x: f"${x:.2f}" if x else "N/A")
            if distribution == "Student-t (Fat tails)" or distribution == "Normal":
                df_display['Lower Bound'] = df_display['Lower Bound'].apply(lambda x: f"${x:.2f}")
                df_display['Upper Bound'] = df_display['Upper Bound'].apply(lambda x: f"${x:.2f}")
                df_display['Within Bounds'] = df_display['Within Bounds'].apply(lambda x: "✓" if x else "✗")
                display_cols = ['Test Date', 'Forecast Date', 'Starting Price', 'Actual Price',
                        'Lower Bound', 'Upper Bound', 'Volatility', 'Within Bounds']
            else:
                 display_cols = ['Test Date', 'Forecast Date', 'Starting Price', 'Actual Price',
                        'Lower Bound Normal', 'Upper Bound Normal', 'Lower Bound Student-t', 'Upper Bound Student-t', 'Volatility', 'Within Bounds Normal', 'Within Bounds Student-t']

            df_display['Volatility'] = df_display['Volatility'].apply(lambda x: f"{x:.1f}%")
            
            # Select columns to display
            st.dataframe(df_display[display_cols], use_container_width=True)

            # Visualization with dark theme
            st.write("---")
            st.subheader("📈 Validation Over Time")

            import matplotlib.pyplot as plt
            import seaborn as sns

            # Configure for dark theme
            fig, ax = plt.subplots(figsize=(14, 7), facecolor='#0e1117')
            ax.set_facecolor('#0e1117')

            # Plot bounds and actual prices (use valid results only)
            dates = pd.to_datetime(df_valid['Test Date'])

            # Confidence interval
            ax.fill_between(dates, df_valid['Lower Bound'], df_valid['Upper Bound'],
                            alpha=0.3, color='#4A90E2', label='90% Confidence Interval', zorder=1)

            # Plot bounds
            ax.plot(dates, df_valid['Lower Bound'], color='#E74C3C', linestyle='--',
                linewidth=2, alpha=0.8, label='Lower Bound (5%)', zorder=2)
            ax.plot(dates, df_valid['Upper Bound'], color='#E74C3C', linestyle='--',
                linewidth=2, alpha=0.8, label='Upper Bound (95%)', zorder=2)

            # Plot actual prices
            hits = df_valid[df_valid['Within Bounds'] == True]
            misses = df_valid[df_valid['Within Bounds'] == False]

            if len(hits) > 0:
                ax.scatter(pd.to_datetime(hits['Test Date']), hits['Actual Price'],
                        color='#27AE60', s=80, zorder=5, label='✓ Within Bounds',
                        marker='o', edgecolors='white', linewidths=1.5, alpha=0.8)
            if len(misses) > 0:
                misses_valid = misses[misses['Actual Price'].notna()]
                if len(misses_valid) > 0:
                    ax.scatter(pd.to_datetime(misses_valid['Test Date']), misses_valid['Actual Price'],
                            color='#E74C3C', s=120, zorder=6, label='✗ Outside Bounds',
                            marker='X', edgecolors='white', linewidths=2, alpha=0.9)

            # Styling
            ax.set_xlabel('Test Date', fontsize=13, fontweight='bold', color='white')
            ax.set_ylabel('Price ($)', fontsize=13, fontweight='bold', color='white')
            ax.set_title(f'Backtest Validation: {ticker} - {timeframe.replace("_", " ").title()}',
                        fontsize=16, fontweight='bold', pad=20, color='white')

            # Legend with dark theme
            legend = ax.legend(loc='upper left', frameon=True, fancybox=True,
                            facecolor='#1e1e1e', edgecolor='#2d2d2d', framealpha=0.95,
                            fontsize=10, markerscale=1.2, labelcolor='white')
            legend.get_frame().set_linewidth(1)

            # Grid
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='#2d2d2d')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#2d2d2d')
            ax.spines['bottom'].set_color('#2d2d2d')

            # Rotate x-axis labels
            plt.xticks(rotation=45, ha='right', color='white')
            plt.yticks(color='white')
            plt.tight_layout(pad=1.0)
            st.pyplot(fig, use_container_width=True)

    else:
        st.error("No valid backtest results. Try different parameters or date range.")