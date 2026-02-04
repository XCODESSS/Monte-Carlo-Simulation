import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta 
from scipy.stats import binomtest
from Monte_Carlo import *
import os
from Calibration import (
    calculate_calibration_metrics,
    calculate_directional_accuracy,
    identify_failure_patterns
)

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

    # Distribution selection
    st.write("---")
    distribution = st.selectbox(
        "Distribution Model",
        ["Normal", "Student-t (Fat Tails)", "Compare Both"],
        help="Choose which distribution to validate"
    )
    
    if distribution == "Student-t (Fat Tails)" or distribution == "Compare Both":
        df = st.slider(
            "Degrees of Freedom (Student-t)",
            min_value=3,
            max_value=30,
            value=7,
            help="Lower values = fatter tails (more extreme events)"
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
    
    # Generate test dates going BACKWARDS from end_date
    test_dates = pd.date_range(end=end_date, periods=num_backtests, freq='ME')
    test_dates = test_dates.sort_values()

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
    progress_bar = st.progress(0)
    status_text = st.empty()

    full_data = cached_download_stock_data(ticker, start_date_str, end_date_str)
    
    for i, test_date in enumerate(test_dates):
        status_text.text(f"Running backtest {i+1}/{num_backtests}...")
        progress_bar.progress((i + 1) / num_backtests)
        mask = full_data.index <= test_date
        data = full_data[mask].copy()
        
        try:
            
            if len(data) < 30:
                continue
            
            
            if use_rolling_window and lookback_days:
                if len(data) > lookback_days:
                    stats_data = data.tail(lookback_days)
                else:
                    stats_data = data
            else:
                stats_data = data

            # Calculate statistics
            stats = calculate_statistics(stats_data)
            
            
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
                
            elif distribution == "Student-t (Fat Tails)":
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
                
            else:  
                
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
                
                # For comparison mode, track both
                lower_bound_normal = metrics_normal['lower_bound']
                upper_bound_normal = metrics_normal['upper_bound']
                lower_bound_student_t = metrics_student_t['lower_bound']
                upper_bound_student_t = metrics_student_t['upper_bound']

            # Get actual price at exactly num_days trading days after test_date
            calendar_days_buffer = int(num_days * 1.4) + 20
            forecast_date = test_date + pd.Timedelta(days=calendar_days_buffer)
            
            # Download data from test_date onwards to get the future price
            actual_data = download_stock_data(
                ticker, 
                test_date.strftime('%Y-%m-%d'), 
                forecast_date.strftime('%Y-%m-%d'),
                api_key=alpha_vantage_key
            )
            
            # Get the price at exactly num_days trading days after test_date
            if len(actual_data) > num_days:
                # Check if test_date is in the data (it's a trading day)
                test_date_in_data = test_date in actual_data.index or test_date.date() in [d.date() for d in actual_data.index]
                
                if test_date_in_data or actual_data.index[0].date() == test_date.date():
                    actual_price = float(actual_data['Close'].iloc[num_days])
                    actual_forecast_date = actual_data.index[num_days]
                else:
                    actual_price = float(actual_data['Close'].iloc[num_days - 1])
                    actual_forecast_date = actual_data.index[num_days - 1]
                    
            elif len(actual_data) >= num_days:
                idx_to_use = min(num_days, len(actual_data) - 1)
                actual_price = float(actual_data['Close'].iloc[idx_to_use])
                actual_forecast_date = actual_data.index[idx_to_use]
            else:
                # Not enough data - skip this test
                actual_price = None
                actual_forecast_date = None

            # Store results
            if distribution == "Compare Both":
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
                    'mu': stats['mu'],
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
        
        # Display based on distribution mode
        if distribution == "Compare Both":
            # Comparison mode display
            st.success("✅ Validation Complete!")
            
            # Calculate hit rates for both models
            hit_rate_normal = (df_valid['Within Bounds Normal'].sum() / len(df_valid)) * 100
            hit_rate_student_t = (df_valid['Within Bounds Student-t'].sum() / len(df_valid)) * 100
            
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
            
            if not df_valid.empty:
                st.write("---")
                st.header("🔬 Calibration Analysis")
                
                # Create normalized DataFrames for calibration (fix column name mismatch)
                base_cols = ['Test Date', 'Starting Price', 'Actual Price', 'Volatility', 'mu']
                
                # Normal model data
                df_normal = df_valid.rename(columns={
                    'Within Bounds Normal': 'Within Bounds',
                    'Lower Bound Normal': 'Lower Bound',
                    'Upper Bound Normal': 'Upper Bound'
                })[base_cols + ['Within Bounds', 'Lower Bound', 'Upper Bound']]
                
                # Student-t model data  
                df_student = df_valid.rename(columns={
                    'Within Bounds Student-t': 'Within Bounds',
                    'Lower Bound Student-t': 'Lower Bound',
                    'Upper Bound Student-t': 'Upper Bound'
                })[base_cols + ['Within Bounds', 'Lower Bound', 'Upper Bound']]
                
                # Calculate metrics for both models separately
                calib_metrics_normal = calculate_calibration_metrics(df_normal)
                calib_metrics_student = calculate_calibration_metrics(df_student)
                

                calib_metrics = calib_metrics_student if 'regime_analysis' in calib_metrics_student else calib_metrics_normal
                
                # Directional accuracy (pass num_days for proper GBM median calculation)
                dir_accuracy_normal = calculate_directional_accuracy(df_normal, num_days=num_days)
                dir_accuracy_student = calculate_directional_accuracy(df_student, num_days=num_days)
                
                # Failure patterns
                failure_patterns_normal = identify_failure_patterns(df_normal)
                failure_patterns_student = identify_failure_patterns(df_student)
                
                # Display regime-based performance
                st.subheader("📊 Performance by Volatility Regime")
                
                regime_data = []
                for regime, stats in calib_metrics['regime_analysis'].items():
                    regime_data.append({
                        'Regime': regime,
                        'Hit Rate': f"{stats['hit_rate']:.1f}%",
                        'Tests': stats['count'],
                        'Avg Miss Magnitude': f"{stats['avg_miss_magnitude']:.1f}%"
                    })
                
                if regime_data:
                    regime_df = pd.DataFrame(regime_data)
                    st.dataframe(regime_df, use_container_width=True, hide_index=True)
                    
                    # Interpretation
                    low_vol_hit = calib_metrics['regime_analysis'].get('Low', {}).get('hit_rate', 0)
                    high_vol_hit = calib_metrics['regime_analysis'].get('High', {}).get('hit_rate', 0)
                    
                    if low_vol_hit > 0 and high_vol_hit > 0:
                        drop = low_vol_hit - high_vol_hit
                        st.info(f"💡 **Key Finding:** Model accuracy drops {drop:.1f}% in high volatility regimes")
                

                if hit_rate_student_t >= hit_rate_normal:
                    calib_metrics = calib_metrics_student
                    dir_accuracy = dir_accuracy_student
                    failure_patterns = failure_patterns_student
                else:
                    calib_metrics = calib_metrics_normal
                    dir_accuracy = dir_accuracy_normal
                    failure_patterns = failure_patterns_normal

                st.write("---")
                st.subheader("🎯 Directional Accuracy")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Overall Direction",
                        f"{dir_accuracy['overall_accuracy']*100:.1f}%",
                        help="% of times we correctly predicted up vs down"
                    )
                with col2:
                    st.metric(
                        "Upward Moves",
                        f"{dir_accuracy['upward_accuracy']*100:.1f}%",
                        help="Accuracy when stock went up"
                    )
                with col3:
                    st.metric(
                        "Downward Moves",
                        f"{dir_accuracy['downward_accuracy']*100:.1f}%",
                        help="Accuracy when stock went down"
                    )
                
                # Failure analysis
                st.write("---")
                st.subheader("⚠️ Failure Case Analysis")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric(
                        "Major Failures (>15% miss)",
                        int(failure_patterns['major_failures']),
                        delta=f"{failure_patterns['failure_rate']:.1f}% of all tests"
                    )
                with col_b:
                    st.metric(
                        "Total Misses",
                        int(failure_patterns['total_misses']),
                        delta=f"{failure_patterns['total_misses']/len(df_valid)*100:.1f}% of tests"
                    )
                
                if failure_patterns['patterns']:
                    st.write("**Failure Patterns:**")
                    for pattern in failure_patterns['patterns']:
                        st.write(f"- {pattern}")
                
                # Worst misses table
                if calib_metrics['worst_misses']:
                    st.write("---")
                    st.subheader("📉 Worst Prediction Misses")
                    
                    worst_df = pd.DataFrame(calib_metrics['worst_misses'])
                    worst_df['Actual Price'] = worst_df['Actual Price'].apply(lambda x: f"${x:.2f}")
                    worst_df['Lower Bound'] = worst_df['Lower Bound'].apply(lambda x: f"${x:.2f}")
                    worst_df['Upper Bound'] = worst_df['Upper Bound'].apply(lambda x: f"${x:.2f}")
                    worst_df['Miss Magnitude'] = worst_df['Miss Magnitude'].apply(lambda x: f"{x:.1f}%")
                    worst_df['Volatility'] = worst_df['Volatility'].apply(lambda x: f"{x:.1f}%")
                    
                    st.dataframe(worst_df, use_container_width=True, hide_index=True)
                    
                    st.caption("*These represent the most significant prediction errors*")
            
            # Calculate interval widths
            df_valid['Interval Width Normal'] = (df_valid['Upper Bound Normal'] - df_valid['Lower Bound Normal']) / df_valid['Starting Price'] * 100
            df_valid['Interval Width Student-t'] = (df_valid['Upper Bound Student-t'] - df_valid['Lower Bound Student-t']) / df_valid['Starting Price'] * 100
            
            avg_width_normal = df_valid['Interval Width Normal'].mean()
            avg_width_student_t = df_valid['Interval Width Student-t'].mean()
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
            
            # Detailed comparison table
            st.write("---")
            st.subheader("📊 Detailed Comparison Table")
            
            comparison_df = pd.DataFrame({
                'Metric': ['Hit Rate', 'Avg Interval Width', 'Hits', 'Misses'],
                'Normal': [
                    f"{hit_rate_normal:.1f}%",
                    f"{avg_width_normal:.1f}%",
                    int(df_valid['Within Bounds Normal'].sum()),
                    int(len(df_valid) - df_valid['Within Bounds Normal'].sum())
                ],
                'Student-t': [
                    f"{hit_rate_student_t:.1f}%",
                    f"{avg_width_student_t:.1f}%",
                    int(df_valid['Within Bounds Student-t'].sum()),
                    int(len(df_valid) - df_valid['Within Bounds Student-t'].sum())
                ]
            })
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            # Visualization comparing both models
            st.write("---")
            st.subheader("📈 Validation Over Time - Both Models")
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), facecolor='#0e1117')
            
            dates = pd.to_datetime(df_valid['Test Date'])
            
            # Normal Distribution Plot
            ax1.set_facecolor('#0e1117')
            ax1.fill_between(dates, df_valid['Lower Bound Normal'], df_valid['Upper Bound Normal'], 
                            alpha=0.3, color='#4A90E2', label='90% Confidence Interval', zorder=1)
            ax1.plot(dates, df_valid['Lower Bound Normal'], color='#4A90E2', linestyle='--', 
                   linewidth=2, alpha=0.8, label='Bounds', zorder=2)
            ax1.plot(dates, df_valid['Upper Bound Normal'], color='#4A90E2', linestyle='--', 
                   linewidth=2, alpha=0.8, zorder=2)
            
            hits_normal = df_valid[df_valid['Within Bounds Normal'] == True]
            misses_normal = df_valid[df_valid['Within Bounds Normal'] == False]
            
            if len(hits_normal) > 0:
                ax1.scatter(pd.to_datetime(hits_normal['Test Date']), hits_normal['Actual Price'], 
                          color='#27AE60', s=80, zorder=5, label='✓ Within Bounds', 
                          marker='o', edgecolors='white', linewidths=1.5, alpha=0.8)
            if len(misses_normal) > 0:
                ax1.scatter(pd.to_datetime(misses_normal['Test Date']), misses_normal['Actual Price'], 
                          color='#E74C3C', s=120, zorder=6, label='✗ Outside Bounds', 
                          marker='X', edgecolors='white', linewidths=2, alpha=0.9)
            
            ax1.set_title(f'Normal Distribution - Hit Rate: {hit_rate_normal:.1f}%', 
                        fontsize=14, fontweight='bold', pad=15, color='white')
            ax1.set_ylabel('Price ($)', fontsize=12, fontweight='bold', color='white')
            ax1.legend(loc='upper left', fontsize=9, facecolor='#1e1e1e', edgecolor='#2d2d2d', 
                      framealpha=0.95, labelcolor='white')
            ax1.grid(True, alpha=0.3, color='#2d2d2d')
            ax1.tick_params(colors='white')
            for spine in ax1.spines.values():
                spine.set_color('#2d2d2d')
            
            # Student-t Distribution Plot
            ax2.set_facecolor('#0e1117')
            ax2.fill_between(dates, df_valid['Lower Bound Student-t'], df_valid['Upper Bound Student-t'], 
                            alpha=0.3, color='#E74C3C', label='90% Confidence Interval', zorder=1)
            ax2.plot(dates, df_valid['Lower Bound Student-t'], color='#E74C3C', linestyle='--', 
                   linewidth=2, alpha=0.8, label='Bounds', zorder=2)
            ax2.plot(dates, df_valid['Upper Bound Student-t'], color='#E74C3C', linestyle='--', 
                   linewidth=2, alpha=0.8, zorder=2)
            
            hits_student = df_valid[df_valid['Within Bounds Student-t'] == True]
            misses_student = df_valid[df_valid['Within Bounds Student-t'] == False]
            
            if len(hits_student) > 0:
                ax2.scatter(pd.to_datetime(hits_student['Test Date']), hits_student['Actual Price'], 
                          color='#27AE60', s=80, zorder=5, label='✓ Within Bounds', 
                          marker='o', edgecolors='white', linewidths=1.5, alpha=0.8)
            if len(misses_student) > 0:
                ax2.scatter(pd.to_datetime(misses_student['Test Date']), misses_student['Actual Price'], 
                          color='#E74C3C', s=120, zorder=6, label='✗ Outside Bounds', 
                          marker='X', edgecolors='white', linewidths=2, alpha=0.9)
            
            ax2.set_title(f'Student-t Distribution (df={df}) - Hit Rate: {hit_rate_student_t:.1f}%', 
                        fontsize=14, fontweight='bold', pad=15, color='white')
            ax2.set_xlabel('Test Date', fontsize=12, fontweight='bold', color='white')
            ax2.set_ylabel('Price ($)', fontsize=12, fontweight='bold', color='white')
            ax2.legend(loc='upper left', fontsize=9, facecolor='#1e1e1e', edgecolor='#2d2d2d', 
                      framealpha=0.95, labelcolor='white')
            ax2.grid(True, alpha=0.3, color='#2d2d2d')
            ax2.tick_params(colors='white')
            for spine in ax2.spines.values():
                spine.set_color('#2d2d2d')
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            
        else:
            # Single model display
            st.success("✅ Backtest Complete!")
            
            st.info("""
            📊 **What is Hit Rate?**

            The model generates a 90% confidence interval (5th to 95th percentile).
            If the model is well-calibrated, actual prices should fall within this range ~90% of the time.

            - **Hit Rate > 85%**: Excellent calibration ✅
            - **Hit Rate 75-85%**: Good calibration ✓
            - **Hit Rate < 75%**: Model may need adjustment ⚠️
            """)
            
            # Calculate hit rate
            hit_rate = (df_valid['Within Bounds'].sum() / len(df_valid)) * 100
            num_hits = df_valid['Within Bounds'].sum()
            num_misses = len(df_valid) - num_hits
            above_upper = (df_valid['Actual Price'] > df_valid['Upper Bound']).sum()
            below_lower = (df_valid['Actual Price'] < df_valid['Lower Bound']).sum()
            
            # Display key metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Hit Rate", f"{hit_rate:.1f}%", 
                        help="Percentage of times actual price fell within 90% confidence interval")
            with col2:
                st.metric("Successful Tests", int(num_hits), 
                        delta=f"out of {len(df_valid)} valid")
            with col3:
                st.metric("Total Tests", len(df_results),
                        delta=f"{len(df_results) - len(df_valid)} skipped")
            
            # Calculate diagnostics
            avg_interval_width = ((df_valid['Upper Bound'] - df_valid['Lower Bound']) / df_valid['Starting Price'] * 100).mean()
            
            n = len(df_valid)
            expected_hit_rate = 90.0
            margin_of_error = 1.96 * np.sqrt(0.9 * 0.1 / n) * 100 if n > 0 else 0
            is_within_expected = abs(hit_rate - expected_hit_rate) <= margin_of_error
            
            p_value = binomtest(int(num_hits), n, 0.9, alternative='two-sided').pvalue if n > 0 else 1.0
            
            # Additional Diagnostics
            st.write("---")
            st.subheader("📊 Diagnostic Information")
            
            diag_col1, diag_col2, diag_col3, diag_col4 = st.columns(4)
            with diag_col1:
                st.metric("Above Upper Bound", int(above_upper), 
                        help="Number of times actual price exceeded 95th percentile")
            with diag_col2:
                st.metric("Below Lower Bound", int(below_lower),
                        help="Number of times actual price fell below 5th percentile")
            with diag_col3:
                st.metric("Avg Interval Width", f"{avg_interval_width:.1f}%",
                        help="Average width of 90% confidence interval as % of starting price")
            with diag_col4:
                st.metric("Expected Range", f"90.0% ± {margin_of_error:.1f}%",
                        help=f"Expected hit rate for {n} samples (95% confidence)")
            
            # Statistical interpretation
            st.write("---")
            st.subheader("📈 Statistical Analysis")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.write("**Statistical Significance:**")
                st.write(f"- P-value (binomial test): {p_value:.4f}")
                if p_value < 0.05:
                    st.warning("⚠️ Statistically significant deviation from 90%")
                else:
                    st.success("✅ Not statistically significant (within expected range)")
                st.write(f"- Margin of error (95% CI): ±{margin_of_error:.1f}%")
                st.write(f"- Your result: {hit_rate:.1f}%")
            
            with col_b:
                st.write("**Model Performance:**")
                if hit_rate >= 85:
                    st.success(f"✅ Excellent! Hit rate of {hit_rate:.1f}% indicates well-calibrated model.")
                elif hit_rate >= 75:
                    st.info(f"✓ Good. Hit rate of {hit_rate:.1f}% is close to expected 90%.")
                else:
                    st.warning(f"⚠️ Hit rate of {hit_rate:.1f}% is below expected. Model may need adjustment.")
            
            # Show misses if any
            if num_misses > 0:
                st.write("---")
                st.subheader("🔍 Sample Misses")
                misses_df = df_valid[~df_valid['Within Bounds']].head(3)
                for idx, row in misses_df.iterrows():
                    with st.expander(f"Miss on {row['Test Date']}"):
                        st.write(f"**Starting Price:** ${row['Starting Price']:.2f}")
                        st.write(f"**Actual Price:** ${row['Actual Price']:.2f}")
                        st.write(f"**Bounds:** [${row['Lower Bound']:.2f}, ${row['Upper Bound']:.2f}]")
                        if row['Actual Price'] < row['Lower Bound']:
                            st.write(f"**Result:** Below lower bound by ${row['Lower Bound'] - row['Actual Price']:.2f}")
                        else:
                            st.write(f"**Result:** Above upper bound by ${row['Actual Price'] - row['Upper Bound']:.2f}")
            
            # Visualization
            st.write("---")
            st.subheader("📈 Validation Over Time")
            
            fig, ax = plt.subplots(figsize=(14, 7), facecolor='#0e1117')
            ax.set_facecolor('#0e1117')
            
            dates = pd.to_datetime(df_valid['Test Date'])
            
            ax.fill_between(dates, df_valid['Lower Bound'], df_valid['Upper Bound'], 
                            alpha=0.3, color='#4A90E2', label='90% Confidence Interval', zorder=1)
            ax.plot(dates, df_valid['Lower Bound'], color='#E74C3C', linestyle='--', 
                   linewidth=2, alpha=0.8, label='Lower Bound (5%)', zorder=2)
            ax.plot(dates, df_valid['Upper Bound'], color='#E74C3C', linestyle='--', 
                   linewidth=2, alpha=0.8, label='Upper Bound (95%)', zorder=2)
            
            hits = df_valid[df_valid['Within Bounds'] == True]
            misses = df_valid[df_valid['Within Bounds'] == False]
            
            if len(hits) > 0:
                ax.scatter(pd.to_datetime(hits['Test Date']), hits['Actual Price'], 
                          color='#27AE60', s=80, zorder=5, label='✓ Within Bounds', 
                          marker='o', edgecolors='white', linewidths=1.5, alpha=0.8)
            if len(misses) > 0:
                ax.scatter(pd.to_datetime(misses['Test Date']), misses['Actual Price'], 
                          color='#E74C3C', s=120, zorder=6, label='✗ Outside Bounds', 
                          marker='X', edgecolors='white', linewidths=2, alpha=0.9)
            
            ax.set_xlabel('Test Date', fontsize=13, fontweight='bold', color='white')
            ax.set_ylabel('Price ($)', fontsize=13, fontweight='bold', color='white')
            ax.set_title(f'Backtest Validation: {ticker} - {timeframe.replace("_", " ").title()}', 
                        fontsize=16, fontweight='bold', pad=20, color='white')
            
            legend = ax.legend(loc='upper left', frameon=True, fancybox=True,
                              facecolor='#1e1e1e', edgecolor='#2d2d2d', framealpha=0.95,
                              fontsize=10, markerscale=1.2, labelcolor='white')
            legend.get_frame().set_linewidth(1)
            
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='#2d2d2d')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#2d2d2d')
            ax.spines['bottom'].set_color('#2d2d2d')
            
            plt.xticks(rotation=45, ha='right', color='white')
            plt.yticks(color='white')
            plt.tight_layout(pad=1.0)
            st.pyplot(fig, use_container_width=True)
            
            # Detailed results table
            st.write("---")
            st.subheader("📋 Detailed Results")
            
            df_display = df_valid.copy()
            df_display['Actual Price'] = df_display['Actual Price'].apply(lambda x: f"${x:.2f}")
            df_display['Lower Bound'] = df_display['Lower Bound'].apply(lambda x: f"${x:.2f}")
            df_display['Upper Bound'] = df_display['Upper Bound'].apply(lambda x: f"${x:.2f}")
            df_display['Within Bounds'] = df_display['Within Bounds'].apply(lambda x: "✓" if x else "✗")
            df_display['Volatility'] = df_display['Volatility'].apply(lambda x: f"{x:.1f}%")
            
            display_cols = ['Test Date', 'Forecast Date', 'Starting Price', 'Actual Price', 
                           'Lower Bound', 'Upper Bound', 'Within Bounds', 'Volatility']
            st.dataframe(df_display[display_cols], use_container_width=True)
        
    else:
        st.error("No valid backtest results. Try different parameters or date range.")