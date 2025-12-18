import streamlit as st
# Import your functions from the refactored code
from Monte_Carlo import *

# Streamlit page config
st.title("📈 Monte Carlo Stock Price Simulator")

# User inputs
ticker = st.text_input("Stock Ticker", value="AAPL")
timeframe = st.selectbox("Forecast Timeframe", list(TIMEFRAMES.keys()))
num_simulations = st.slider("Number of Simulations", min_value=100, max_value=10000, value=1000, step=100)

# Run button
if st.button("Run Simulation"):
    # Your code to run simulation and display results goes here
    
    data = download_stock_data(ticker, '2015-12-14', '2025-12-14')

    # Step 2: Calculate statistics  
    stats = calculate_statistics(data)  # Returns dict with mean, std, starting_price

    # Step 3: Get num_days from timeframe
    num_days = TIMEFRAMES[timeframe]

    # Step 4: Run simulation
    simulations = run_monte_carlo(stats['starting_price'], stats['mean'], stats['std'], num_days, num_simulations)

    # Step 5: Calculate metrics
    metrics = calculate_metrics(simulations, stats['starting_price'], stats['mean'], stats['std'], num_days)
    # Step 6: Display results
    st.subheader("Simulation Results")
    st.write(f"Projected Price after {timeframe.replace('_', ' ')}:")   
    st.write(f"Mean Final Price: ${metrics['mean_final_price']:.2f}")
    st.write(f"5th Percentile Price: ${metrics['lower_bound']:.2f}")
    st.write(f"95th Percentile Price: ${metrics['upper_bound']:.2f}")
    st.write(f"Average Loss: ${metrics['avg_loss']:.2f}")
    st.write(f"Average Gain: ${metrics['avg_gain']:.2f}")
    st.write(f"Probability of Loss: {metrics['prob_loss']:.2f}%")
    st.write(f"Probability of Gain: {metrics['prob_profit']:.2f}%")
    
    # Visualization
    fig = plot_simulation(simulations, metrics, stats['starting_price'], stats['std'], ticker, num_days, num_simulations)
    st.pyplot(fig)

