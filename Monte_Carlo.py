"""
Monte Carlo Stock Price Simulation
Properly structured with functions and modular design
"""

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# CONFIGURATION

# Timeframe presets (trading days)
TIMEFRAMES = {
    '2_days': 2,
    '1_week': 5,
    '2_weeks': 10,
    '1_month': 20,
    '3_months': 60,
    '6_months': 120,
    '1_year': 252
}


# DATA FUNCTIONS


def download_stock_data(ticker, start_date, end_date):
    """
    Download historical stock data
    
    Args:
        ticker: Stock symbol (e.g., 'AAPL')
        start_date: Start date string (e.g., '2015-12-14')
        end_date: End date string (e.g., '2025-12-14')
    
    Returns:
        DataFrame with stock data
    """
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    return data


def calculate_statistics(data):
    """
    Calculate mean return and volatility from historical data
    
    Args:
        data: DataFrame with stock prices
    
    Returns:
        dict with mean, std, and starting_price
    """
    # Calculate daily returns
    data['Log Return'] = np.log(data['Close'] / data['Close'].shift(1))
    mu = data['Log Return'].mean()
    sigma = data['Log Return'].std()

    
    # Get statistics
    mean = mu
    std = sigma
    starting_price = float(data['Close'].iloc[-1])
    
    return {
        'mean': mean,
        'std': std,
        'starting_price': starting_price
    }


# SIMULATION FUNCTIONS


def run_monte_carlo(starting_price, mu, sigma, num_days, num_simulations):
    dt = 1  # one trading day
    random_shocks = np.random.normal(0, 1, (num_simulations, num_days))
    
    price_paths = np.zeros_like(random_shocks)
    price_paths[:, 0] = starting_price

    for t in range(1, num_days):
        price_paths[:, t] = price_paths[:, t-1] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * random_shocks[:, t]
        )

    return price_paths



# ANALYSIS FUNCTIONS


def calculate_metrics(simulations, starting_price, mean, std, num_days):
    """
    Calculate all risk metrics from simulation results
    
    Args:
        simulations: 2D array of simulation results
        starting_price: Initial stock price
        mean: Average daily return
        std: Daily volatility
        num_days: Number of days simulated
    
    Returns:
        dict with all metrics
    """
    # Get final prices
    final_prices = simulations[:, -1]
    
    # Basic statistics
    mean_final_price = np.mean(final_prices)
    lower_bound = np.percentile(final_prices, 5)
    upper_bound = np.percentile(final_prices, 95)
    
    # Value at Risk
    var_95_price = np.percentile(final_prices, 5)
    var_95_loss = starting_price - var_95_price
    var_99_price = np.percentile(final_prices, 1)
    var_99_loss = starting_price - var_99_price
    
    # Sharpe Ratio
    annual_return = mu * 252
    annual_volatility = sigma * np.sqrt(252)
    sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0

    
    # Probability of profit/loss
    prob_profit = (final_prices > starting_price).sum() / len(final_prices) * 100
prob_loss = (final_prices < starting_price).sum() / len(final_prices) * 100

# Average gain/loss
returns = (final_prices - starting_price) / starting_price
avg_gain = returns[returns > 0].mean()
avg_loss = returns[returns < 0].mean()

    
    return {
        'mean_final_price': mean_final_price,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'var_95_loss': var_95_loss,
        'var_99_loss': var_99_loss,
        'sharpe_ratio': sharpe_ratio,
        'prob_profit': prob_profit,
        'prob_loss': prob_loss,
        'avg_gain': avg_gain,
        'avg_loss': avg_loss
    }

# VISUALIZATION FUNCTIONS


def plot_simulation(simulations, metrics, starting_price, std, ticker, num_days, num_simulations):
    """
    Create visualization of simulation results
    
    Args:
        simulations: 2D array of simulation paths
        metrics: Dictionary of calculated metrics
        starting_price: Initial price
        std: Daily volatility
        ticker: Stock symbol
        num_days: Number of days simulated
        num_simulations: Number of simulations run
    """
    plt.style.use('bmh')
    fig = plt.figure(figsize=(15, 8), dpi=100)
    gs = gridspec.GridSpec(1, 4, wspace=0.05)
    
    # LEFT PLOT: Simulation Paths
    ax1 = plt.subplot(gs[0, :3])
    days_array = np.arange(1, num_days + 1)
    
    # Plot all paths
    for i in range(len(simulations)):
        ax1.plot(days_array, simulations[i], color='green', alpha=0.05, linewidth=2, zorder=1)
    
    # Confidence cone
    percentiles = np.percentile(simulations, [5, 50, 95], axis=0)
    ax1.fill_between(days_array, percentiles[0], percentiles[2], 
                     color='deepskyblue', alpha=0.15, label='90% Confidence Interval')
    ax1.plot(days_array, percentiles[1], color='dodgerblue', linewidth=2.5, 
            label='Median Projection', zorder=3)
    
    # Starting price reference
    ax1.axhline(y=starting_price, color='black', linestyle='--', 
               linewidth=1.5, alpha=0.5, label='Start Price')
    
    ax1.set_title(f"Monte Carlo Simulation: {ticker} ({num_simulations} runs)", 
                 fontsize=16, fontweight='bold', pad=15)
    ax1.set_ylabel('Stock Price ($)', fontsize=12)
    ax1.set_xlabel('Trading Days', fontsize=12)
    ax1.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
    ax1.margins(x=0)
    
    # RIGHT PLOT: Distribution
    ax2 = plt.subplot(gs[0, 3], sharey=ax1)
    final_prices = simulations[:, -1]
    
    ax2.hist(final_prices, bins=20, orientation='horizontal', 
            color='dodgerblue', alpha=0.6, edgecolor='white')
    ax2.axhline(y=metrics['mean_final_price'], color='navy', linestyle='-', linewidth=1)
    ax2.axhline(y=metrics['lower_bound'], color='red', linestyle=':', linewidth=1.5)
    ax2.axhline(y=metrics['upper_bound'], color='red', linestyle=':', linewidth=1.5)
    
    plt.setp(ax2.get_yticklabels(), visible=False)
    ax2.set_xlabel('Freq', fontsize=10)
    ax2.set_title('Distribution', fontsize=12)
    
    # INFO BOX
    stats_text = (
        f"Start Price:   ${starting_price:.2f}\n"
        f"Mean Final:    ${metrics['mean_final_price']:.2f}\n"
        f"VaR (5%):     ${metrics['lower_bound']:.2f}\n"
        f"Upside (95%):  ${metrics['upper_bound']:.2f}\n"
        f"Volatility:    {std*100:.2f}%"
    )
    ax1.text(0.02, 0.02, stats_text, transform=ax1.transAxes, fontsize=11,
            verticalalignment='bottom', 
            bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.9, ec='lightgrey'))
    
    plt.tight_layout()
    return fig


def print_results(ticker, metrics, num_days):
    """
    Print formatted results to console
    
    Args:
        ticker: Stock symbol
        metrics: Dictionary of calculated metrics
        num_days: Number of days simulated
    """
    print(f"\n{'='*60}")
    print(f"MONTE CARLO SIMULATION RESULTS: {ticker}")
    print(f"{'='*60}")
    print(f"Forecast Period: {num_days} trading days")
    print(f"\nPRICE PREDICTIONS:")
    print(f"  Mean Final Price:        ${metrics['mean_final_price']:.2f}")
    print(f"  90% Confidence Interval: [${metrics['lower_bound']:.2f}, ${metrics['upper_bound']:.2f}]")
    print(f"\nRISK METRICS:")
    print(f"  Value at Risk (95%):     ${metrics['var_95_loss']:.2f}")
    print(f"  Value at Risk (99%):     ${metrics['var_99_loss']:.2f}")
    print(f"  Sharpe Ratio:            {metrics['sharpe_ratio']:.2f}")
    print(f"\nPROBABILITIES:")
    print(f"  Probability of Profit:   {metrics['prob_profit']:.1f}%")
    print(f"  Probability of Loss:     {metrics['prob_loss']:.1f}%")
    print(f"  Average Gain:            ${metrics['avg_gain']:.2f}")
    print(f"  Average Loss:            ${metrics['avg_loss']:.2f}")
    print(f"{'='*60}\n")


# MAIN EXECUTION


def main():
    """
    Main function to run the Monte Carlo simulation
    """
    # Configuration
    ticker = 'AAPL'
    start_date = '2015-12-14'
    end_date = '2025-12-14'
    timeframe = '1_year'  # Change this to any key from TIMEFRAMES
    num_simulations = 1000
    
    # Get number of days from timeframe
    num_days = TIMEFRAMES[timeframe]
    
    print(f"Downloading data for {ticker}...")
    data = download_stock_data(ticker, start_date, end_date)
    
    print("Calculating statistics...")
    stats = calculate_statistics(data)
    
    print(f"Running {num_simulations} Monte Carlo simulations for {num_days} days...")
    simulations = run_monte_carlo(
        stats['starting_price'],
        stats['mean'],
        stats['std'],
        num_days,
        num_simulations
    )
    
    print("Analyzing results...")
    metrics = calculate_metrics(
        simulations,
        stats['starting_price'],
        stats['mean'],
        stats['std'],
        num_days
    )
    
    # Display results
    print_results(ticker, metrics, num_days)
    
    # Visualize
    plot_simulation(
        simulations,
        metrics,
        stats['starting_price'],
        stats['std'],
        ticker,
        num_days,
        num_simulations
    )


# Run the simulation
if __name__ == "__main__":
    main()