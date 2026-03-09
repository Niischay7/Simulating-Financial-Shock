import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

assets = {
    'S&P 500': '^GSPC',
    'NASDAQ': '^IXIC',
    'Gold': 'GC=F',
    '10-Yr Treasury': 'ZN=F' # Replaced TLT with Treasury Note Futures
}
weights = np.array([0.25, 0.25, 0.25, 0.25])
start_date = '2000-01-01' # Adjusted start date for ZN=F availability
shock_start_date = '2001-09-17'
shock_end_date = '2001-09-28'
end_date = '2002-12-31'

data = yf.download(list(assets.values()), start=start_date, end=end_date)['Close']
returns = data.pct_change().dropna()

pre_shock_returns = returns[:'2001-09-10']
shock_returns = returns[shock_start_date:shock_end_date]

mean_returns = pre_shock_returns.mean().values
cov_matrix = pre_shock_returns.cov()
cholesky_L = np.linalg.cholesky(cov_matrix)

portfolio_pre_shock_returns = (pre_shock_returns * weights).sum(axis=1)
mu_p = portfolio_pre_shock_returns.mean()
sigma_p = portfolio_pre_shock_returns.std()
jump_threshold = 3 * sigma_p
jumps = portfolio_pre_shock_returns[abs(portfolio_pre_shock_returns) > jump_threshold]
lambda_j = len(jumps) / len(portfolio_pre_shock_returns)
mu_j = jumps.mean()
sigma_j = jumps.std()


num_simulations = 1000
prediction_days = 252
num_assets = len(assets)
dt = 1


def run_portfolio_simulation(start_returns=None):
    all_sim_returns = np.zeros((prediction_days, num_simulations))
    
    for i in range(num_simulations):
        sim_returns = []
        #begin with any prev shocks if given else proceed with normal sim
        if start_returns is not None and not start_returns.empty:
            sim_returns = (start_returns * weights).sum(axis=1).tolist()

        days_to_sim = prediction_days - len(sim_returns)
        
        uncorrelated_normals = np.random.normal(0, 1, (days_to_sim, num_assets))
        correlated_normals = uncorrelated_normals @ cholesky_L.T
        
        for k in range(days_to_sim):

            drift = (mean_returns - 0.5 * np.diag(cov_matrix)) * dt
            diffusion = correlated_normals[k] * np.sqrt(dt)
            
            # portfolio-level jump 
            jump = 0
            if np.random.poisson(lambda_j * dt) > 0:
                jump = np.random.normal(mu_j, sigma_j)
            
            asset_returns = np.exp(drift + diffusion) - 1
            portfolio_return = np.sum(asset_returns * weights) + jump
            sim_returns.append(portfolio_return)
            
        all_sim_returns[:, i] = (1 + np.array(sim_returns)).cumprod()
        
    return all_sim_returns


jump_diffusion_sims = run_portfolio_simulation()
shock_sims = run_portfolio_simulation(start_returns=shock_returns)

final_jump_returns = jump_diffusion_sims[-1, :] - 1
final_shock_returns = shock_sims[-1, :] - 1

def calculate_drawdown(simulations):
    cumulative_returns = np.vstack([np.ones(simulations.shape[1]), simulations])
    running_max = np.maximum.accumulate(cumulative_returns, axis=0)
    drawdowns = (cumulative_returns - running_max) / running_max
    return np.min(drawdowns, axis=0)

jump_drawdowns = calculate_drawdown(jump_diffusion_sims)
shock_drawdowns = calculate_drawdown(shock_sims)

def calculate_risk_metrics(final_returns, drawdowns, confidence_level=0.95):
    var = np.percentile(final_returns, (1 - confidence_level) * 100)
    cvar = final_returns[final_returns <= var].mean()
    return {
        "VaR (95%)": var, "CVaR (95%)": cvar, "Volatility": np.std(final_returns),
        "Skewness": skew(final_returns), "Kurtosis": kurtosis(final_returns),
        "Avg Final Return": np.mean(final_returns), "Avg Max Drawdown": np.mean(drawdowns)
    }

jump_metrics = calculate_risk_metrics(final_jump_returns, jump_drawdowns)
shock_metrics = calculate_risk_metrics(final_shock_returns, shock_drawdowns)

##we look at option pricing during shock pricing
strike_price = 1.0  #
time_to_maturity_days = 90 
risk_free_rate = 0.02 


portfolio_price_at_maturity = shock_sims[time_to_maturity_days - 1, :]
#here i assumed european call option 
call_option_payoffs = np.maximum(portfolio_price_at_maturity - strike_price, 0)

time_to_maturity_years = time_to_maturity_days / 252
average_payoff = np.mean(call_option_payoffs)
call_option_price = average_payoff * np.exp(-risk_free_rate * time_to_maturity_years)



plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Portfolio Simulation: Impact of 9/11 Market Shock', fontsize=16)
axes[0, 0].plot(jump_diffusion_sims[:, :100], color='blue', alpha=0.1)
axes[0, 0].set_title('Portfolio Jump-Diffusion Simulations')
axes[0, 1].plot(shock_sims[:, :100], color='red', alpha=0.1)
axes[0, 1].set_title('Portfolio Shock-Integrated Simulations')
axes[1, 0].hist(final_jump_returns, bins=50, alpha=0.7, label='Jump-Diffusion', color='blue')
axes[1, 0].hist(final_shock_returns, bins=50, alpha=0.7, label='Shock', color='red')
axes[1, 0].set_title('Distribution of Final Portfolio Returns')
axes[1, 0].legend()
axes[1, 1].hist(jump_drawdowns, bins=50, alpha=0.7, label='Jump-Diffusion', color='blue')
axes[1, 1].hist(shock_drawdowns, bins=50, alpha=0.7, label='Shock', color='red')
axes[1, 1].set_title('Distribution of Max Portfolio Drawdowns')
axes[1, 1].legend()
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('portfolio_shock_simulation.png')
plt.show()


print("\n" + "="*50)
print("Portfolio Analysis: Jump-Diffusion vs. Shock Scenario")
print("="*50)
metrics_df = pd.DataFrame({'Jump-Diffusion': jump_metrics, 'Shock': shock_metrics})
print(metrics_df.to_string(formatters={
    'Jump-Diffusion': '{:,.2%}'.format, 'Shock': '{:,.2%}'.format
}))
print("\n" + "="*50)
print("Portfolio Jump Parameters (Pre-Shock Period):")
print(f"  - Jump Intensity (λ): {lambda_j:.4f} (jumps per day)")
print(f"  - Mean Jump Size (μ_j): {mu_j:.4f}")
print(f"  - Jump Volatility (σ_j): {sigma_j:.4f}")
print("="*50)

print("\n" + "="*50)
print("European Call Option Pricing (Post-Shock)")
print("="*50)
print(f"  - Time to Maturity: {time_to_maturity_days} days")
print(f"  - Strike Price: {strike_price:.2f} (At-the-money)")
print(f"  - Assumed Risk-Free Rate: {risk_free_rate:.2%}")
print(f"  - Estimated Call Option Price: ${call_option_price:.4f}")
print("="*50)