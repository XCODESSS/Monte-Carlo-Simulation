import streamlit as st
import pandas as pd
from Monte_Carlo import *

st.title("📊 Model Validation Dashboard")
st.write("Backtest the Monte Carlo model on historical data to verify accuracy")

# User inputs
ticker = st.text_input("Stock Ticker", value="AAPL")
timeframe = st.selectbox("Forecast Timeframe", list(TIMEFRAMES.keys()))
num_backtests = st.slider("Number of Backtests", min_value=10, max_value=100, value=30, step=5)

if st.button("Run Backtest", type="primary"):
    num_days = TIMEFRAMES[timeframe]
    end_date = pd.Timestamp('2023-01-01')  # Fixed date to ensure we have future data
    test_dates = pd.date_range(end=end_date, periods=num_backtests, freq='M')
    
    results = []
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, test_date in enumerate(test_dates):
        status_text.text(f"Running backtest {i+1}/{num_backtests}...")
        progress_bar.progress((i + 1) / num_backtests)
        
        try:
            # Download data UP TO test_date only
            data = download_stock_data(ticker, '2015-01-01', test_date.strftime('%Y-%m-%d'))
            
            if len(data) < 30:  # Skip if insufficient data
                continue
            
            # Calculate statistics
            stats = calculate_statistics(data)
            
            # Run simulation
            simulations = run_monte_carlo(
                stats['starting_price'], 
                stats['mean'], 
                stats['std'], 
                num_days, 
                1000
            )

            # Calculate metrics
            metrics = calculate_metrics(
                simulations, 
                stats['starting_price'], 
                stats['mean'], 
                stats['std'], 
                num_days
            )
            lower_bound = metrics['lower_bound']
            upper_bound = metrics['upper_bound']

            # Get actual price at forecast_date
            forecast_date = test_date + pd.Timedelta(days=num_days + 10)  # Extra buffer
            actual_data = download_stock_data(
                ticker, 
                test_date.strftime('%Y-%m-%d'), 
                forecast_date.strftime('%Y-%m-%d')
            )
            
            
            
            # Get the price at approximately num_days trading days later
            if len(actual_data) > num_days:
                actual_price = float(actual_data['Close'].iloc[num_days])
                actual_within_bounds = lower_bound <= actual_price <= upper_bound
            else:
                actual_price = None
                actual_within_bounds = False

            # Store result
            validation_result = {
                'Test Date': test_date.strftime('%Y-%m-%d'),
                'Forecast Date': forecast_date.strftime('%Y-%m-%d'),
                'Actual Price': actual_price,
                'Lower Bound': lower_bound,
                'Upper Bound': upper_bound,
                'Within Bounds': actual_within_bounds
            }

            results.append(validation_result)

           
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
        
        # Calculate hit rate
        hit_rate = (df_results['Within Bounds'].sum() / len(df_results)) * 100
        
        # Display summary
        st.success("Backtest Complete!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Hit Rate", f"{hit_rate:.1f}%", 
                     help="Percentage of times actual price fell within 90% confidence interval")
        with col2:
            st.metric("Successful Tests", df_results['Within Bounds'].sum())
        with col3:
            st.metric("Total Tests", len(df_results))
        
        # Interpretation
        st.write("---")
        st.subheader("Interpretation")
        if hit_rate >= 85:
            st.success(f"✅ Excellent! The model's 90% confidence intervals captured actual prices {hit_rate:.1f}% of the time. This suggests the model is well-calibrated.")
        elif hit_rate >= 75:
            st.info(f"✓ Good. The model captured actual prices {hit_rate:.1f}% of the time. Close to the expected 90%.")
        else:
            st.warning(f"⚠️ The model only captured actual prices {hit_rate:.1f}% of the time. This is below the expected 90%, suggesting the model may need adjustment.")
        
        # Display detailed results
        st.write("---")
        st.subheader("Detailed Results")
        
        # Format the dataframe for display
        df_display = df_results.copy()
        df_display['Actual Price'] = df_display['Actual Price'].apply(lambda x: f"${x:.2f}" if x else "N/A")
        df_display['Lower Bound'] = df_display['Lower Bound'].apply(lambda x: f"${x:.2f}")
        df_display['Upper Bound'] = df_display['Upper Bound'].apply(lambda x: f"${x:.2f}")
        df_display['Within Bounds'] = df_display['Within Bounds'].apply(lambda x: "✓" if x else "✗")
        
        st.dataframe(df_display, use_container_width=True)
        
        # Visualization
        st.write("---")
        st.subheader("Validation Over Time")
        
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot bounds and actual prices
        dates = pd.to_datetime(df_results['Test Date'])
        ax.plot(dates, df_results['Lower Bound'], 'r--', label='Lower Bound (5%)', alpha=0.7)
        ax.plot(dates, df_results['Upper Bound'], 'r--', label='Upper Bound (95%)', alpha=0.7)
        ax.fill_between(dates, df_results['Lower Bound'], df_results['Upper Bound'], 
                        alpha=0.2, color='blue', label='90% Confidence Interval')
        
        # Plot actual prices (color by success/failure)
        hits = df_results[df_results['Within Bounds'] == True]
        misses = df_results[df_results['Within Bounds'] == False]
        
        if len(hits) > 0:
            ax.scatter(pd.to_datetime(hits['Test Date']), hits['Actual Price'], 
                      color='green', s=50, zorder=5, label='Hit', marker='o')
        if len(misses) > 0 and not misses['Actual Price'].isna().all():
            ax.scatter(pd.to_datetime(misses['Test Date']), misses['Actual Price'], 
                      color='red', s=50, zorder=5, label='Miss', marker='x')
        
        ax.set_xlabel('Test Date', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.set_title(f'Backtest Results: {ticker} - {timeframe.replace("_", " ")}', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        st.pyplot(fig)
        
    else:
        st.error("No valid backtest results. Try different parameters or date range.")