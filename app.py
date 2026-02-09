import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from Monte_Carlo import *
from datetime import datetime, timedelta
import os

def cached_download_stock_data(ticker, start_date, end_date):
    """Cached wrapper for download_stock_data"""
    return download_stock_data(ticker, start_date, end_date)


st.set_page_config(layout="wide", page_title="Monte Carlo Stock Simulator")


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

st.title(" Monte Carlo Stock Price Simulator")


with st.sidebar:
    st.header("Configuration")
    
    ticker = st.text_input("Stock Ticker", value="AAPL")
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
    
    # Days selection
    st.subheader("📅 Forecast Period")
    use_custom_days = st.checkbox("Use custom number of days")
    
    if use_custom_days:
        num_days = st.number_input(
            "Number of Trading Days",
            min_value=1,
            max_value=365,
            value=20
        )
        timeframe_label = f"{num_days} days"
    else:
        timeframe = st.selectbox("Forecast Timeframe", list(TIMEFRAMES.keys()))
        num_days = TIMEFRAMES[timeframe]
        timeframe_label = timeframe.replace('_', ' ')
    
    # Distribution selection
    st.subheader("📊 Distribution Model")
    
    compare_models = st.checkbox(
        "🔀 Compare Normal vs Student-t Models",
        value=False,
        help="Run both models side-by-side for comparison"
    )
    
    if compare_models:
        df = st.slider(
            "Degrees of Freedom (Student-t)",
            min_value=3,
            max_value=30,
            value=7,
            help="Lower values = fatter tails (more extreme events)"
        )
        distribution = "Both"
    else:
        distribution = st.selectbox(
            "Return Distribution", 
            ["Normal", "Student-t (Fat Tails)"]
        )
        
        if distribution == "Student-t (Fat Tails)":
            df = st.slider(
                "Degrees of Freedom",
                min_value=3,
                max_value=30,
                value=7,
                help="Lower values = fatter tails (more extreme events)"
            )
        else:
            df = None
    
  
    # Simulations
    st.subheader("🔢 Simulations")
    num_simulations = st.slider(
        "Number of Simulations",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100
    )
    
    st.divider()
    run_button = st.button("🚀 Run Simulation", type="primary", use_container_width=True)

if run_button:

    with st.spinner("Downloading stock data..."):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 10)

        try:
            data = cached_download_stock_data(
                ticker, 
                start_date.strftime('%Y-%m-%d'), 
                end_date.strftime('%Y-%m-%d')
            )
            
            if data.empty:
                st.error(f"❌ No data found for ticker '{ticker}'.")
                st.stop()
        
        except Exception as e:
            st.error(f"❌ Error downloading data: {str(e)}")
            st.info("💡 Tip: Use a valid stock ticker (e.g., AAPL, MSFT, GOOGL)")
            st.stop()
    
   
   
    # Calculate statistics
    with st.spinner("Calculating statistics..."):
        stats = calculate_statistics(data)

    # Run simulation based on distribution choice
    if distribution == "Both":
        with st.spinner(f"Running {num_simulations} simulations for both models..."):
            simulations_normal = run_monte_carlo(
                stats["starting_price"],
                stats["mu"],
                stats["sigma"],
                num_days,
                num_simulations
            )
            simulations_student_t = run_monte_carlo_student_t(
                stats["starting_price"],
                stats["mu"],
                stats["sigma"],
                num_days,
                num_simulations,
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
                    num_simulations
                )
            else:

                simulations = run_monte_carlo_student_t(
                    stats["starting_price"],
                    stats["mu"],
                    stats["sigma"],
                    num_days,
                    num_simulations,
                    df=df
                )

        metrics = calculate_metrics(
            simulations, 
            stats['starting_price'], 
            stats['mu'], 
            stats['sigma'], 
            num_days
        )
    
    
    
    # Display results
    st.subheader("Simulation Results")

    if use_custom_days:
        timeframe_label = f"{num_days} days"
    else:
        timeframe_label = timeframe.replace('_', ' ')
    st.write(f"**Ticker:** {ticker} | **Period:** {timeframe_label} | **Distribution:** {distribution}")
    
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Current Price",
            f"${stats['starting_price']:.2f}"
        )

    with col2:
        price_change = metrics['mean_final_price'] - stats['starting_price']
        st.metric(
            "Expected Price",
            f"${metrics['mean_final_price']:.2f}",
            f"{price_change:+.2f} ({price_change/stats['starting_price']*100:+.1f}%)"
        )

    with col3:
        st.metric(
            "Value at Risk (95%)",
            f"${metrics['var_95_loss']:.2f}",
            delta_color="inverse"
        )

    with col4:
        st.metric(
            "Probability of Profit",
            f"{metrics['prob_profit']:.1f}%"
        )
    
    st.divider()
    col_a, col_b = st.columns(2)
    
    with col_a:
        if metrics['avg_loss'] < 0:
            st.write(f"📉 **Average Loss:** {metrics['avg_loss']*100:.2f}%")
        st.write(f"**Probability of Loss:** {metrics['prob_loss']:.1f}%")
    
    with col_b:
        if metrics['avg_gain'] > 0:
            st.write(f"📈 **Average Gain:** {metrics['avg_gain']*100:.2f}%")
        st.write(f"**Probability of Profit:** {metrics['prob_profit']:.1f}%")
    
    
    # Risk-adjusted performance
    st.divider()
    st.subheader("Risk-Adjusted Performance")
    st.write(f"**Sharpe Ratio:** {metrics['sharpe_ratio_post']:.2f}")

    if metrics['sharpe_ratio_post'] > 2:
        st.success("Excellent risk-adjusted returns")
    elif metrics['sharpe_ratio_post'] > 1:
        st.info("Good risk-adjusted returns")
    else:
        st.warning("Low risk-adjusted returns")
    
    
    
    # Visualization
    if distribution == "Both":
        try:
            _ = simulations_normal
            _ = simulations_student_t
            _ = metrics_normal
            _ = metrics_student_t
        except NameError as e:
            st.error(f"❌ Error: Comparison variables not properly initialized. Please try running the simulation again.")
            st.error(f"Details: {str(e)}")
            st.stop()

        col1, col2 = st.columns(2)
        
        days_array = np.arange(1, num_days + 1)
        num_paths = min(200, len(simulations_normal))

        with col1:
            fig1 = plt.figure(figsize=(8, 6), facecolor='#0e1117', dpi=100)
            ax1 = fig1.add_subplot(111)

            for i in range(num_paths):
                ax1.plot(days_array, simulations_normal[: num_paths].T, color='#4A90E2', alpha=0.05, linewidth=0.5, zorder=1)
            
            percentiles_norm = np.percentile(simulations_normal, [5, 50, 95], axis=0)
            # Distinct shaded area for 90% CI
            ax1.fill_between(days_array, percentiles_norm[0], percentiles_norm[2], 
                             color='#4A90E2', alpha=0.3, label='90% CI', zorder=2)
            # Thick, solid line for Median
            ax1.plot(days_array, percentiles_norm[1], color='#FFFFFF', linewidth=3, 
                    label='Median', zorder=4)
            ax1.axhline(y=stats['starting_price'], color='#FFD700', linestyle='--', 
                       linewidth=2, alpha=0.8, label='Start Price', zorder=3)
            
            ax1.set_title('Normal Distribution (GBM)', fontsize=14, fontweight='bold', pad=10, color='white')
            ax1.set_ylabel('Price ($)', fontsize=11, fontweight='bold', color='white')
            ax1.set_xlabel('Trading Days', fontsize=11, fontweight='bold', color='white')
            ax1.legend(loc='upper left', fontsize=9, facecolor='#1e1e1e', edgecolor='#2d2d2d', 
                      framealpha=0.95, labelcolor='white')
            ax1.grid(True, alpha=0.3, color='#2d2d2d')
            ax1.set_facecolor('#0e1117')
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.spines['left'].set_color('#2d2d2d')
            ax1.spines['bottom'].set_color('#2d2d2d')
            plt.tight_layout()
            st.pyplot(fig1, use_container_width=True)
        
        # Student-t Distribution Plot
        with col2:
            fig2 = plt.figure(figsize=(8, 6), facecolor='#0e1117', dpi=100)
            ax2 = fig2.add_subplot(111)
            
            # Faint cloud of paths (very low alpha for clarity)
            for i in range(num_paths):
                ax2.plot(days_array, simulations_student_t[: num_paths].T, color='#E74C3C', alpha=0.05, linewidth=0.5, zorder=1)
            
            percentiles_t = np.percentile(simulations_student_t, [5, 50, 95], axis=0)
            # Distinct shaded area for 90% CI
            ax2.fill_between(days_array, percentiles_t[0], percentiles_t[2], 
                            color='#E74C3C', alpha=0.3, label='90% CI', zorder=2)
            # Thick, solid line for Median
            ax2.plot(days_array, percentiles_t[1], color='#FFFFFF', linewidth=3, 
                    label='Median', zorder=4)
            ax2.axhline(y=stats['starting_price'], color='#FFD700', linestyle='--', 
                       linewidth=2, alpha=0.8, label='Start Price', zorder=3)
            
            ax2.set_title(f'Student-t Distribution (df={df})', fontsize=14, fontweight='bold', pad=10, color='white')
            ax2.set_ylabel('Price ($)', fontsize=11, fontweight='bold', color='white')
            ax2.set_xlabel('Trading Days', fontsize=11, fontweight='bold', color='white')
            ax2.legend(loc='upper left', fontsize=9, facecolor='#1e1e1e', edgecolor='#2d2d2d', 
                      framealpha=0.95, labelcolor='white')
            ax2.grid(True, alpha=0.3, color='#2d2d2d')
            ax2.set_facecolor('#0e1117')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['left'].set_color('#2d2d2d')
            ax2.spines['bottom'].set_color('#2d2d2d')
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)
        
        # Full-width distribution comparison at bottom
        st.subheader("📊 Distribution Comparison")
        fig3 = plt.figure(figsize=(14, 5), facecolor='#0e1117', dpi=100)
        ax3 = fig3.add_subplot(111)
        
        final_prices_norm = simulations_normal[:, -1]
        final_prices_t = simulations_student_t[:, -1]
        
        # Use seaborn histplot with step element and KDE for clean comparison
        sns.histplot(final_prices_norm, bins=40, alpha=0.4, color='#4A90E2', 
                    label=f'Normal (Mean: ${metrics_normal["mean_final_price"]:.2f})', 
                    element='step', kde=True, ax=ax3, edgecolor='#4A90E2', linewidth=1.5)
        sns.histplot(final_prices_t, bins=40, alpha=0.4, color='#E74C3C', 
                    label=f'Student-t (Mean: ${metrics_student_t["mean_final_price"]:.2f})', 
                    element='step', kde=True, ax=ax3, edgecolor='#E74C3C', linewidth=1.5)
        
        # Confidence interval lines
        ax3.axvline(metrics_normal['lower_bound'], color='#4A90E2', linestyle='--', linewidth=1.5, alpha=0.6)
        ax3.axvline(metrics_normal['upper_bound'], color='#4A90E2', linestyle='--', linewidth=1.5, alpha=0.6)
        ax3.axvline(metrics_student_t['lower_bound'], color='#E74C3C', linestyle='--', linewidth=1.5, alpha=0.6)
        ax3.axvline(metrics_student_t['upper_bound'], color='#E74C3C', linestyle='--', linewidth=1.5, alpha=0.6)
        
        ax3.set_xlabel('Final Price ($)', fontsize=12, fontweight='bold', color='white')
        ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold', color='white')
        ax3.set_title('Final Price Distribution Comparison', fontsize=13, fontweight='bold', pad=10, color='white')
        ax3.legend(loc='upper right', fontsize=10, facecolor='#1e1e1e', edgecolor='#2d2d2d', 
                  framealpha=0.95, labelcolor='white')
        ax3.grid(True, alpha=0.3, axis='y', color='#2d2d2d')
        ax3.set_facecolor('#0e1117')
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.spines['left'].set_color('#2d2d2d')
        ax3.spines['bottom'].set_color('#2d2d2d')
        plt.tight_layout()
        st.pyplot(fig3, use_container_width=True)
        
        # Comparison metrics table
        st.write("---")
        st.subheader("📊 Model Comparison Metrics")
        
        comp_col1, comp_col2, comp_col3, comp_col4 = st.columns(4)
        with comp_col1:
            st.metric("Lower Bound (5%)", 
                     f"Normal: ${metrics_normal['lower_bound']:.2f}\nStudent-t: ${metrics_student_t['lower_bound']:.2f}")
        with comp_col2:
            st.metric("Upper Bound (95%)", 
                     f"Normal: ${metrics_normal['upper_bound']:.2f}\nStudent-t: ${metrics_student_t['upper_bound']:.2f}")
        with comp_col3:
            interval_width_norm = (metrics_normal['upper_bound'] - metrics_normal['lower_bound']) / stats['starting_price'] * 100
            interval_width_t = (metrics_student_t['upper_bound'] - metrics_student_t['lower_bound']) / stats['starting_price'] * 100
            st.metric("Interval Width", 
                     f"Normal: {interval_width_norm:.1f}%\nStudent-t: {interval_width_t:.1f}%")
        with comp_col4:
            st.metric("VaR (95%)", 
                     f"Normal: ${metrics_normal['var_95_loss']:.2f}\nStudent-t: ${metrics_student_t['var_95_loss']:.2f}")
        
        st.info("💡 **Key Insight**: Student-t distribution typically shows wider confidence intervals due to fat tails, capturing more extreme events but being more conservative.")
    else:
        fig = plot_simulation(
            simulations, 
            metrics, 
            stats['starting_price'], 
            stats['sigma'], 
            ticker, 
            num_days, 
            num_simulations
        )
        st.pyplot(fig, use_container_width=True)

