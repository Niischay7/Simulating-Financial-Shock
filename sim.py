import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#we do the data retrieval from S&P 500
data = yf.download('^GSPC', start='1995-01-01', end='2002-12-31')
data['Returns'] = data['Close'].pct_change()
pre_shock_data = data[:'2001-09-10']
shock_data = data['2001-09-17':'2001-09-28'] #market reopened after 8 days 

#we have to define the pre-attack period to split the data.
mu = pre_shock_data['Returns'].mean()#avg
sigma = pre_shock_data['Returns'].std()#deviation

num_simulations = 1000
prediction_days = 252 # 1 trading year

#this is for the normal simulation (where attack didnt happen maybe in another universe)

#we simulate 1000 possible paths
normal_simulations = np.zeros((prediction_days, num_simulations))
for i in range(num_simulations):
    daily_returns = np.random.normal(mu, sigma, prediction_days)
    normal_simulations[:, i] = (1 + daily_returns).cumprod()

#we introduce attack which unfortunately happened
shock_returns = shock_data['Returns'].values
shock_simulations = np.zeros((prediction_days, num_simulations))
for i in range(num_simulations):

    shock_period_len = len(shock_returns)
    simulated_returns = np.random.normal(mu, sigma, prediction_days - shock_period_len)
    
    # combine both
    full_returns = np.concatenate([shock_returns, simulated_returns])
    shock_simulations[:, i] = (1 + full_returns).cumprod()

#now we4 anlyze the results
final_normal_returns = normal_simulations[-1, :] - 1
final_shock_returns = shock_simulations[-1, :] - 1

# Calculate drawdowns
def calculate_drawdown(simulations):
    cumulative_returns = np.vstack([np.ones(simulations.shape[1]), simulations])
    running_max = np.maximum.accumulate(cumulative_returns, axis=0)
    drawdowns = (cumulative_returns - running_max) / running_max
    return np.min(drawdowns, axis=0)

normal_drawdowns = calculate_drawdown(normal_simulations)
shock_drawdowns = calculate_drawdown(shock_simulations)


plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Monte Carlo Simulation: Impact of 9/11 Market Shock', fontsize=16)
axes[0, 0].plot(normal_simulations[:, :100], color='blue', alpha=0.1)
axes[0, 0].set_title('Normal Market Simulations')
axes[0, 0].set_ylabel('Cumulative Returns')

axes[0, 1].plot(shock_simulations[:, :100], color='red', alpha=0.1)
axes[0, 1].set_title('Shock-Integrated Simulations')
axes[1, 0].hist(final_normal_returns, bins=50, alpha=0.7, label='Normal', color='blue')
axes[1, 0].hist(final_shock_returns, bins=50, alpha=0.7, label='Shock', color='red')
axes[1, 0].set_title('Distribution of Final Returns (1 Year)')
axes[1, 0].set_xlabel('Final Return')
axes[1, 0].legend()

axes[1, 1].hist(normal_drawdowns, bins=50, alpha=0.7, label='Normal', color='blue')
axes[1, 1].hist(shock_drawdowns, bins=50, alpha=0.7, label='Shock', color='red')
axes[1, 1].set_title('Distribution of Maximum Drawdowns')
axes[1, 1].set_xlabel('Maximum Drawdown')
axes[1, 1].legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('financial_shock_simulation.png')
plt.show()


print("Summary:")
print(f"Average Final Return (Normal): {np.mean(final_normal_returns):.2%}")
print(f"Average Final Return (Shock): {np.mean(final_shock_returns):.2%}")
print("-" * 30)
print(f"Average Max Drawdown (Normal): {np.mean(normal_drawdowns):.2%}")
print(f"Average Max Drawdown (Shock): {np.mean(shock_drawdowns):.2%}")