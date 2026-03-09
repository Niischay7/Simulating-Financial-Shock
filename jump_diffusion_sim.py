import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis


data = yf.download('^GSPC', start='1995-01-01', end='2002-12-31')

data['Returns'] = data['Close'].pct_change()

pre_shock_data = data[:'2001-09-10'].dropna()
shock_data = data['2001-09-17':'2001-09-28'] 

mu = pre_shock_data['Returns'].mean()
sigma = pre_shock_data['Returns'].std()


jump_threshold = 3 * sigma 
jumps = pre_shock_data['Returns'][abs(pre_shock_data['Returns']) > jump_threshold]
lambda_j = len(jumps) / len(pre_shock_data) 
mu_j = jumps.mean() # Mean jump size
sigma_j = jumps.std() # Jump volatility


num_simulations = 1000
prediction_days = 252 
dt = 1 


jump_diffusion_simulations = np.zeros((prediction_days, num_simulations))
start_price = 1 

for i in range(num_simulations):
    prices = [start_price]
    for _ in range(prediction_days):
        # gbm
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * np.random.normal(0, 1)
        
        # Jump 
        poisson_proc = np.random.poisson(lambda_j * dt)
        jump = 0
        if poisson_proc > 0:
            # in case of jump
            jump = np.sum(np.random.normal(mu_j, sigma_j, poisson_proc))

        total_return = drift + diffusion + jump
        next_price = prices[-1] * np.exp(total_return)
        prices.append(next_price)
    
    jump_diffusion_simulations[:, i] = np.array(prices[1:]) / start_price


shock_simulations = np.zeros((prediction_days, num_simulations))
for i in range(num_simulations):
    shock_period_len = len(shock_data['Returns'])
    

    prices = (1 + shock_data['Returns']).cumprod().tolist()
    
    for _ in range(prediction_days - shock_period_len):
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * np.random.normal(0, 1)
        poisson_proc = np.random.poisson(lambda_j * dt)
        jump = 0
        if poisson_proc > 0:
            jump = np.sum(np.random.normal(mu_j, sigma_j, poisson_proc))
        
        total_return = drift + diffusion + jump
        next_price = prices[-1] * np.exp(total_return)
        prices.append(next_price)

    shock_simulations[:, i] = np.array(prices) / prices[0]



final_jump_returns = jump_diffusion_simulations[-1, :] - 1
final_shock_returns = shock_simulations[-1, :] - 1


def calculate_drawdown(simulations):
    cumulative_returns = np.vstack([np.ones(simulations.shape[1]), simulations])
    running_max = np.maximum.accumulate(cumulative_returns, axis=0)
    drawdowns = (cumulative_returns - running_max) / running_max

    return np.min(drawdowns, axis=0)

jump_drawdowns = calculate_drawdown(jump_diffusion_simulations)
shock_drawdowns = calculate_drawdown(shock_simulations)

# 5. Risk Metrics Calculation
def calculate_risk_metrics(final_returns, drawdowns, confidence_level=0.95):

    var = np.percentile(final_returns, (1 - confidence_level) * 100)
    cvar = final_returns[final_returns <= var].mean()
    volatility = np.std(final_returns)
    skewness = skew(final_returns)
    kurt = kurtosis(final_returns) 

    return {
        "VaR (95%)": var,
        "CVaR (95%)": cvar,
        "Volatility": volatility,
        "Skewness": skewness,
        "Kurtosis": kurt,
        "Avg Final Return": np.mean(final_returns),
        "Avg Max Drawdown": np.mean(drawdowns)
    }

jump_diffusion_metrics = calculate_risk_metrics(final_jump_returns, jump_drawdowns)
shock_metrics = calculate_risk_metrics(final_shock_returns, shock_drawdowns)


# 6. Visualization
plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Merton Jump-Diffusion Simulation: Impact of 9/11 Market Shock', fontsize=16)

axes[0, 0].plot(jump_diffusion_simulations[:, :100], color='blue', alpha=0.1)
axes[0, 0].set_title('Jump-Diffusion Market Simulations')
axes[0, 0].set_ylabel('Cumulative Returns')

axes[0, 1].plot(shock_simulations[:, :100], color='red', alpha=0.1)
axes[0, 1].set_title('Shock-Integrated Simulations')

axes[1, 0].hist(final_jump_returns, bins=50, alpha=0.7, label='Jump-Diffusion', color='blue')
axes[1, 0].hist(final_shock_returns, bins=50, alpha=0.7, label='Shock', color='red')
axes[1, 0].set_title('Distribution of Final Returns (1 Year)')
axes[1, 0].set_xlabel('Final Return')
axes[1, 0].legend()

axes[1, 1].hist(jump_drawdowns, bins=50, alpha=0.7, label='Jump-Diffusion', color='blue')
axes[1, 1].hist(shock_drawdowns, bins=50, alpha=0.7, label='Shock', color='red')
axes[1, 1].set_title('Distribution of Maximum Drawdowns')
axes[1, 1].set_xlabel('Maximum Drawdown')
axes[1, 1].legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('financial_shock_jump_diffusion_simulation.png')
plt.show()

# 7. Summary and Risk Metrics
print("Analysis Results: Jump-Diffusion vs. Shock Scenario")
metrics_df = pd.DataFrame({
    'Jump-Diffusion': jump_diffusion_metrics,
    'Shock': shock_metrics
})
metrics_df.index.name = 'Metric'

# Format the output for readability
pd.options.display.float_format = '{:,.4f}'.format
print(metrics_df.to_string(formatters={
    'Jump-Diffusion': '{:,.2%}'.format,
    'Shock': '{:,.2%}'.format
}))

print("\n" + "="*50)
print("Estimated Jump Parameters (Pre-Shock Period):")
print(f"  - Jump Intensity (λ): {lambda_j:.4f} (jumps per day)")
print(f"  - Mean Jump Size (μ_j): {mu_j:.4f}")
print(f"  - Jump Volatility (σ_j): {sigma_j:.4f}")
print("="*50)