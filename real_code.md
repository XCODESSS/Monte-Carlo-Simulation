import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ticker = 'AAPL'
data = yf.download(ticker, start='2022-06-01', end='2023-01-06')

#print(data.shape)
#print(data.head())
today_prices = data.loc['2022-06-08','Close']
yesterday_prices = data.loc['2022-06-07','Close']
#print(today_prices)
#print(yesterday_prices)
daily_ret = (today_prices - yesterday_prices) / yesterday_prices
daily_ret_per = daily_ret \* 100
#print(daily_ret_per)
#print(daily_ret)

data['Daily Return'] = (data['Close'] - data['Close'].shift(1)) / data['Close'].shift(1)
#print(data[['Close', 'Daily Return']])
mean = data['Daily Return'].mean()
std = data['Daily Return'].std()
#print(f'Mean Daily Return: {mean}')
#print(f'Standard Deviation of Daily Return: {std}')

# Get starting price

starting_price = data['Close'].iloc[-1]

# Create array to store simulations

simulations = np.zeros((100, 5)) # 100 simulations, 5 days each

# Your loops go here

for i in range(100):
price = starting_price
for j in range(5):
daily_returns = np.random.normal(mean, std)
price = price \*(1 + daily_returns)
simulations[i][j] = price
#print(simulations)
final_prices = simulations[:, -1]
mean_final_price = np.mean(final_prices)
min = final_prices.min()
max = final_prices.max()
print(f'Mean final price after 5 days: {mean_final_price}')
#print(f'Minimum final price after 5 days: {min}')
#print(f'Maximum final price after 5 days: {max}'

# Assuming you have: simulations, starting_price, final_prices, mean, std, ticker

# from your existing code

# Calculate confidence intervals

lower_bound = np.percentile(final_prices, 5)
upper_bound = np.percentile(final_prices, 95)
mean_final = np.mean(final_prices)

# Create figure with better size

plt.figure(figsize=(14, 8))

# Plot all simulation paths

days = range(1, simulations.shape[1] + 1)
for i in range(simulations.shape[0]):
plt.plot(days, simulations[i], color='steelblue', alpha=0.05, linewidth=2)

# Plot mean path

mean_path = simulations.mean(axis=0)
plt.plot(days, mean_path, color='red', linewidth=2.5, label='Mean Path', zorder=5)

# Add starting price line

starting_price_value = starting_price.iloc[0]
plt.axhline(y=starting_price_value, color='green', linestyle='--', linewidth=2,
label=f'Starting Price: ${starting_price_value:.2f}', alpha=0.8)

# Add confidence interval boundaries

plt.axhline(y=lower_bound, color='orange', linestyle='--', linewidth=2,
label=f'5th Percentile: ${lower_bound:.2f}', alpha=0.8)
plt.axhline(y=upper_bound, color='orange', linestyle='--', linewidth=2,
label=f'95th Percentile: ${upper_bound:.2f}', alpha=0.8)

# Formatting

plt.title(f'Monte Carlo Simulation: {ticker} Stock Price Forecast\n{simulations.shape[0]} Simulations over {simulations.shape[1]} Trading Days',
fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Trading Days', fontsize=13, fontweight='bold')
plt.ylabel('Stock Price ($)', fontsize=13, fontweight='bold')
plt.legend(loc='best', fontsize=11, framealpha=0.9)
plt.grid(True, alpha=0.3, linestyle='--')

# Add statistics text box

stats_text = f'Mean Final Price: ${mean_final:.2f}\n90% CI: [${lower_bound:.2f}, ${upper_bound:.2f}]\nVolatility: {std\*100:.2f}%'
plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
fontsize=11, verticalalignment='top',
bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.show()

# Monte Carlo Simulation for Stock Price Prediction

max_price = simulations.max()
min_price = simulations.min()
avg_price = simulations.mean()
realistic_price = (max_price + min_price) / 2

#print(f'Starting price: {starting_price}')
#print(f'Realistic price after 5 days: {realistic_price}')
real_price_after_5_days = data['Close'].iloc[-1 + 5]
#print(f'Actual price after 5 days: {real_price_after_5_days}')
lower_bound = np.percentile(final_prices, 5)
upper_bound = np.percentile(final_prices, 95)
print(f'90% confidence interval for the final price after 5 days: [{lower_bound}, {upper_bound}]')
