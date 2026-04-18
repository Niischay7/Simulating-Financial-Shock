import yfinance as yf
import pandas as pd
import numpy as np
import sys
import time
import matplotlib.pyplot as plt

# ANSI formatting
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Historical Shock definitions
SHOCKS = {
    '1': {
        'name': '2008 Financial Crisis (Lehman Bankruptcy)',
        'start_date': '2006-01-01',
        'shock_start_date': '2008-09-01',
        'shock_end_date': '2008-11-30',
        'end_date': '2009-12-31'
    },
    '2': {
        'name': 'COVID-19 Global Pandemic Crash',
        'start_date': '2018-01-01',
        'shock_start_date': '2020-02-20',
        'shock_end_date': '2020-03-23',
        'end_date': '2021-12-31'
    },
    '3': {
        'name': '9/11 Terrorist Attacks',
        'start_date': '2000-01-01',
        'shock_start_date': '2001-09-17',
        'shock_end_date': '2001-09-28',
        'end_date': '2002-12-31'
    },
    '4': {
        'name': 'Dot-Com Bubble Burst',
        'start_date': '1998-01-01',
        'shock_start_date': '2000-03-10',
        'shock_end_date': '2002-10-09',
        'end_date': '2003-12-31'
    }
}

def print_banner():
    banner = f"""
{Colors.OKCYAN}{Colors.BOLD}====================================================
      FINANCIAL SHOCK PORTFOLIO SIMULATOR CLI
===================================================={Colors.ENDC}
Stress-test your investments against historical black swans.
    """
    print(banner)

def get_input(prompt, cast_type=str, default=None):
    prompt_str = f"{Colors.OKGREEN}?{Colors.ENDC} {prompt}"
    if default is not None:
        prompt_str += f" [{default}]: "
    else:
        prompt_str += ": "
    
    while True:
        try:
            val = input(prompt_str).strip()
            if not val and default is not None:
                return cast_type(default)
            if not val:
                print(f"{Colors.FAIL}Value cannot be empty.{Colors.ENDC}")
                continue
            return cast_type(val)
        except ValueError:
            print(f"{Colors.FAIL}Invalid input. Please try again.{Colors.ENDC}")

def simulate():
    print_banner()
    
    portfolio_value = get_input("Enter your total portfolio value in USD", float, 100000)
    print(f"\n{Colors.OKBLUE}--- Portfolio Allocation ---{Colors.ENDC}")
    print("Add your assets via Yahoo Finance tickers (e.g. AAPL, SPY, GC=F). Leave blank to finish.")
    
    portfolio = []
    while True:
        ticker = input(f"{Colors.OKGREEN}?{Colors.ENDC} Asset Ticker: ").strip().upper()
        if not ticker:
            if len(portfolio) > 0:
                break
            else:
                print(f"{Colors.WARNING}Please enter at least one asset.{Colors.ENDC}")
                continue
        
        weight = get_input(f"Weight/Allocation % for {ticker}", float, 100 if len(portfolio) == 0 else 0)
        if weight > 0:
            portfolio.append({'ticker': ticker, 'weight': weight})
            
    # Normalize weights
    total_weight = sum([p['weight'] for p in portfolio])
    for p in portfolio:
        p['weight'] = p['weight'] / total_weight

    print(f"\n{Colors.OKBLUE}--- Select Historical Tragedy (Shock Event) ---{Colors.ENDC}")
    for k, v in SHOCKS.items():
        print(f"[{k}] {v['name']}")
    
    while True:
        choice = get_input("Select Event ID", str, "1")
        if choice in SHOCKS:
            shock_config = SHOCKS[choice]
            break
        print(f"{Colors.FAIL}Invalid choice.{Colors.ENDC}")

    print(f"\n{Colors.BOLD}Fetching historical data for {len(portfolio)} asset(s)...{Colors.ENDC}")
    tickers = [p['ticker'] for p in portfolio]
    weights = np.array([p['weight'] for p in portfolio])
    num_assets = len(tickers)

    try:
        data = yf.download(tickers, start=shock_config['start_date'], end=shock_config['end_date'], progress=False)['Close']
        if num_assets == 1:
            data = pd.DataFrame(data)
            data.columns = tickers
    except Exception as e:
        print(f"{Colors.FAIL}Error downloading data: {e}{Colors.ENDC}")
        sys.exit(1)

    returns = data.pct_change().dropna()
    
    shock_start = pd.to_datetime(shock_config['shock_start_date'])
    # adjust timezone if needed
    if returns.index.tz is not None:
        shock_start = shock_start.tz_localize(returns.index.tz)
    shock_end = pd.to_datetime(shock_config['shock_end_date'])
    if returns.index.tz is not None:
        shock_end = shock_end.tz_localize(returns.index.tz)

    try:
        pre_shock_returns = returns.loc[:shock_start - pd.Timedelta(days=1)]
        shock_returns = returns.loc[shock_start:shock_end]
    except Exception:
        pre_shock_returns = returns.iloc[:int(len(returns)*0.8)]
        shock_returns = returns.iloc[int(len(returns)*0.8):]

    if len(pre_shock_returns) == 0:
        print(f"{Colors.FAIL}Not enough historical data for these tickers before the shock date.{Colors.ENDC}")
        sys.exit(1)

    print(f"{Colors.BOLD}Calculating correlations and jump-diffusion parameters...{Colors.ENDC}")
    mean_returns = pre_shock_returns.mean().values
    cov_matrix = pre_shock_returns.cov()
    
    if cov_matrix.isnull().values.any() or (cov_matrix == 0).all().all():
        cholesky_L = np.eye(num_assets)
    else:
        try:
            cholesky_L = np.linalg.cholesky(cov_matrix)
        except np.linalg.LinAlgError:
            cholesky_L = np.eye(num_assets) * np.sqrt(np.diag(cov_matrix) + 1e-8)

    portfolio_pre_shock_returns = (pre_shock_returns * weights).sum(axis=1)
    sigma_p = portfolio_pre_shock_returns.std()
    
    if pd.isna(sigma_p) or sigma_p == 0: sigma_p = 1e-6
        
    jump_threshold = 3 * sigma_p
    jumps = portfolio_pre_shock_returns[abs(portfolio_pre_shock_returns) > jump_threshold]
    lambda_j = len(jumps) / len(portfolio_pre_shock_returns) if len(portfolio_pre_shock_returns) > 0 else 0
    mu_j = jumps.mean() if len(jumps) > 0 else 0
    sigma_j = jumps.std() if len(jumps) > 0 else 0
    
    if pd.isna(sigma_j): sigma_j = 0
    if pd.isna(mu_j): mu_j = 0

    num_simulations = 500
    prediction_days = 252 # 1 Trading Year
    dt = 1
    
    print(f"{Colors.BOLD}Running {num_simulations} Monte Carlo simulation paths ({prediction_days} days projection)...{Colors.ENDC}")

    def run_portfolio_simulation(start_returns=None):
        all_sim_returns = np.zeros((prediction_days, num_simulations))
        for i in range(num_simulations):
            sim_returns = []
            if start_returns is not None and not start_returns.empty:
                sim_returns = (start_returns * weights).sum(axis=1).tolist()
            
            days_to_sim = prediction_days - len(sim_returns)
            if days_to_sim < 0:
                days_to_sim = 0
                sim_returns = sim_returns[:prediction_days]
                
            if days_to_sim > 0:
                uncorrelated_normals = np.random.normal(0, 1, (days_to_sim, num_assets))
                if cholesky_L.shape == (num_assets, num_assets):
                    correlated_normals = uncorrelated_normals @ cholesky_L.T
                else:
                    correlated_normals = uncorrelated_normals
                    
                for k in range(days_to_sim):
                    drift = (mean_returns - 0.5 * np.diag(cov_matrix)) * dt
                    diffusion = correlated_normals[k] * np.sqrt(dt)
                    jump = 0
                    if np.random.poisson(abs(lambda_j) * dt) > 0:
                        jump = np.random.normal(mu_j, sigma_j if sigma_j > 0 else 1e-6)
                    
                    asset_returns = np.exp(drift + diffusion) - 1
                    portfolio_return = np.sum(asset_returns * weights) + jump
                    sim_returns.append(portfolio_return)
                
            sim_np = np.array(sim_returns)
            sim_np = np.nan_to_num(sim_np, nan=0.0, posinf=0.0, neginf=0.0)
            all_sim_returns[:, i] = (1 + sim_np).cumprod()
            
        return all_sim_returns

    jump_sims = run_portfolio_simulation()
    shock_sims = run_portfolio_simulation(start_returns=shock_returns)
    
    final_jump = jump_sims[-1, :] - 1
    final_shock = shock_sims[-1, :] - 1

    def calculate_drawdown(simulations):
        cumulative_returns = np.vstack([np.ones(simulations.shape[1]), simulations])
        running_max = np.maximum.accumulate(cumulative_returns, axis=0)
        drawdowns = (cumulative_returns - running_max) / running_max
        return np.min(drawdowns, axis=0)

    jump_drawdowns = calculate_drawdown(jump_sims)
    shock_drawdowns = calculate_drawdown(shock_sims)

    def print_metric(label, norm_val, shock_val, format_as_currency=False):
        fmt = lambda x: f"${x:,.2f}" if format_as_currency else f"{(x*100):.2f}%"
        nv = fmt(norm_val)
        sv = fmt(shock_val)
        diff = shock_val - norm_val
        diff_fmt = lambda x: f"${x:,.2f}" if format_as_currency else f"{(x*100):.2f}%"
        
        diff_str = diff_fmt(diff)
        if diff > 0 and not diff_str.startswith("-"): diff_str = "+" + diff_str
        
        color = Colors.OKGREEN if diff >= 0 else Colors.FAIL
        print(f"{label:<25} | Normal: {nv:>12} | Shock: {sv:>12} | Delta: {color}{diff_str:>12}{Colors.ENDC}")

    time.sleep(0.5)
    print(f"\n{Colors.OKCYAN}{Colors.BOLD}===================================================={Colors.ENDC}")
    print(f"{Colors.OKCYAN}{Colors.BOLD}                  SIMULATION RESULTS                {Colors.ENDC}")
    print(f"{Colors.OKCYAN}{Colors.BOLD}===================================================={Colors.ENDC}")
    print(f"Tragedy Simulated: {Colors.WARNING}{shock_config['name']}{Colors.ENDC}")
    print(f"Portfolio Initial Value: {Colors.BOLD}${portfolio_value:,.2f}{Colors.ENDC}\n")

    avg_final_norm = np.mean(final_jump)
    avg_final_shock = np.mean(final_shock)
    
    var_norm = np.percentile(final_jump, 5)
    var_shock = np.percentile(final_shock, 5)
    
    mdd_norm = np.mean(jump_drawdowns)
    mdd_shock = np.mean(shock_drawdowns)

    print_metric("Expected Return", avg_final_norm, avg_final_shock)
    print_metric("Value at Risk (95%)", var_norm, var_shock)
    print_metric("Average Max Drawdown", mdd_norm, mdd_shock)
    print("-" * 70)
    print_metric("Proj. Final Balance ($)", portfolio_value * (1 + avg_final_norm), portfolio_value * (1 + avg_final_shock), True)
    print_metric("95% Worst Case Bal ($)", portfolio_value * (1 + var_norm), portfolio_value * (1 + var_shock), True)

    print(f"\n{Colors.BOLD}Summary:{Colors.ENDC}")
    impact = (portfolio_value * (1 + avg_final_shock)) - (portfolio_value * (1 + avg_final_norm))
    if impact < 0:
        print(f"If the '{shock_config['name']}' happened today, your portfolio would theoretically lose an additional {Colors.FAIL}${abs(impact):,.2f}{Colors.ENDC} compared to normal market conditions, taking an additional {Colors.FAIL}{abs(mdd_shock - mdd_norm)*100:.1f}%{Colors.ENDC} max drawdown.")
    else:
        print(f"If the '{shock_config['name']}' happened today, your portfolio would theoretically outperform by {Colors.OKGREEN}${abs(impact):,.2f}{Colors.ENDC} compared to normal conditions.")

    print(f"\n{Colors.OKBLUE}{Colors.BOLD}--- Risk Mitigation & Alternative Strategies ---{Colors.ENDC}")
    print(f"How you could have mitigated this loss and protected your capital:")

    # Calculate Stop-Loss Strategies
    sl_10_finals = []
    trailing_sl_10_finals = []
    
    for i in range(num_simulations):
        path = shock_sims[:, i]
        
        # 1. Strict -10% Stop-Loss
        below_90_idx = np.where(path < 0.90)[0]
        if len(below_90_idx) > 0:
            sl_10_finals.append(path[below_90_idx[0]]) # exit at the price it crossed -10%
        else:
            sl_10_finals.append(path[-1])
            
        # 2. Trailing -10% Stop-Loss (from highest high)
        running_max = np.maximum.accumulate(path)
        drawdown = (path - running_max) / running_max
        below_10_dd_idx = np.where(drawdown < -0.10)[0]
        if len(below_10_dd_idx) > 0:
            trailing_sl_10_finals.append(path[below_10_dd_idx[0]])
        else:
            trailing_sl_10_finals.append(path[-1])
            
    tsl_10_balance = portfolio_value * np.mean(trailing_sl_10_finals)
    shock_balance = portfolio_value * (1 + avg_final_shock)

    # Strategy: Trailing Stop-Loss
    print(f"\n1. {Colors.BOLD}10% Trailing Stop-Loss:{Colors.ENDC}")
    if tsl_10_balance > shock_balance:
        print(f"   By automating a sell order if your portfolio ever drops 10% from its peak,")
        print(f"   your shock balance would average {Colors.OKGREEN}${tsl_10_balance:,.2f}{Colors.ENDC}.")
        print(f"   You would have saved {Colors.OKGREEN}${tsl_10_balance - shock_balance:,.2f}{Colors.ENDC} during the crash.")
    else:
        print(f"   A trailing stop-loss would yield a final balance of ${tsl_10_balance:,.2f}.")

    # Strategy: Hedging with Options
    print(f"\n2. {Colors.BOLD}Hedging with Put Options (Insurance):{Colors.ENDC}")
    cost_of_hedge = portfolio_value * 0.02 # Approx 2% premium for 1-year ATM put protection
    fully_hedged_balance = portfolio_value - cost_of_hedge # Assuming ATM put caps loss at exactly the premium paid
    if fully_hedged_balance > shock_balance:
         print(f"   If you had paid ~2% upfront (${cost_of_hedge:,.2f}) for At-The-Money Put Options,")
         print(f"   your total downside risk is capped, leaving you with at least {Colors.OKGREEN}${fully_hedged_balance:,.2f}{Colors.ENDC}.")
         print(f"   You would have saved {Colors.OKGREEN}${fully_hedged_balance - shock_balance:,.2f}{Colors.ENDC}.")

    # Strategy: Diversification (Cash/Gold)
    print(f"\n3. {Colors.BOLD}Diversification into Safe Havens:{Colors.ENDC}")
    print(f"   Holding 20-30% in un-correlated assets (like Gold or Treasury Bonds) natively")
    print(f"   dampens extreme shock volatility, buying you time to make rational decisions.")

    print(f"\n{Colors.OKCYAN}===================================================={Colors.ENDC}\n")

    # === GRAPHICAL ANALYSIS ===
    print(f"{Colors.BOLD}Generating advanced graphical analysis...{Colors.ENDC}")
    try:
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"Portfolio Risk Analysis Dashboard: {shock_config['name']}", fontsize=18, fontweight='bold')
        
        # --- Plot 1 (Top Left): Average Paths & Stop-Loss ---
        mean_jump = np.mean(jump_sims, axis=1) * portfolio_value
        p5_jump = np.percentile(jump_sims, 5, axis=1) * portfolio_value
        p95_jump = np.percentile(jump_sims, 95, axis=1) * portfolio_value
        
        axs[0, 0].plot(mean_jump, label="Normal - Avg Expected", color='blue', linewidth=2)
        axs[0, 0].fill_between(range(prediction_days), p5_jump, p95_jump, color='blue', alpha=0.1, label="Normal - 90% Confidence")
        
        mean_shock = np.mean(shock_sims, axis=1) * portfolio_value
        p5_shock = np.percentile(shock_sims, 5, axis=1) * portfolio_value
        p95_shock = np.percentile(shock_sims, 95, axis=1) * portfolio_value
        
        axs[0, 0].plot(mean_shock, label="Shock - Avg Expected", color='red', linewidth=2)
        axs[0, 0].fill_between(range(prediction_days), p5_shock, p95_shock, color='red', alpha=0.1, label="Shock - 90% Confidence")
        
        trailing_sl_paths = []
        for i in range(num_simulations):
            path = shock_sims[:, i]
            running_max = np.maximum.accumulate(path)
            drawdown = (path - running_max) / running_max
            below_10 = np.where(drawdown < -0.10)[0]
            
            tsl_path = path.copy()
            if len(below_10) > 0:
                idx = below_10[0]
                tsl_path[idx:] = tsl_path[idx]
            trailing_sl_paths.append(tsl_path)
            
        mean_tsl = np.mean(np.array(trailing_sl_paths), axis=0) * portfolio_value
        axs[0, 0].plot(mean_tsl, label="Shock w/ 10% Trailing Stop", color='green', linewidth=2, linestyle='--')
        
        axs[0, 0].set_title("1. Portfolio Value Projection")
        axs[0, 0].set_xlabel("Trading Days")
        axs[0, 0].set_ylabel("Portfolio Value ($)")
        axs[0, 0].legend(loc="upper left")
        axs[0, 0].grid(True, alpha=0.3)
        ticks = axs[0, 0].get_yticks()
        axs[0, 0].set_yticks(ticks)
        axs[0, 0].set_yticklabels(['${:,.0f}'.format(x) for x in ticks])
        
        # --- Plot 2 (Top Right): Drawdown Over Time ---
        def get_avg_drawdown(sims):
            cumulative_returns = np.vstack([np.ones(sims.shape[1]), sims])
            running_max = np.maximum.accumulate(cumulative_returns, axis=0)
            drawdowns = (cumulative_returns - running_max) / running_max
            return np.mean(drawdowns, axis=1)[1:]

        avg_dd_jump = get_avg_drawdown(jump_sims)
        avg_dd_shock = get_avg_drawdown(shock_sims)
        
        axs[0, 1].plot(avg_dd_jump * 100, color='blue', label="Normal Avg Drawdown", linewidth=2)
        axs[0, 1].plot(avg_dd_shock * 100, color='red', label="Shock Avg Drawdown", linewidth=2)
        axs[0, 1].set_title("2. Average Maximum Drawdown Over Time")
        axs[0, 1].set_xlabel("Trading Days")
        axs[0, 1].set_ylabel("Drawdown (%)")
        axs[0, 1].legend(loc="lower left")
        axs[0, 1].grid(True, alpha=0.3)
        
        # --- Plot 3 (Bottom Left): Final Value Distribution (Histogram) ---
        final_vals_jump = jump_sims[-1, :] * portfolio_value
        final_vals_shock = shock_sims[-1, :] * portfolio_value
        
        axs[1, 0].hist(final_vals_jump, bins=40, alpha=0.5, color='blue', label='Normal Outcomes')
        axs[1, 0].hist(final_vals_shock, bins=40, alpha=0.5, color='red', label='Shock Outcomes')
        axs[1, 0].axvline(np.mean(final_vals_jump), color='blue', linestyle='dashed', linewidth=2)
        axs[1, 0].axvline(np.mean(final_vals_shock), color='red', linestyle='dashed', linewidth=2)
        axs[1, 0].set_title("3. Distribution of Final Portfolio Balances (Year End)")
        axs[1, 0].set_xlabel("Final Balance ($)")
        axs[1, 0].set_ylabel("Number of Simulations")
        axs[1, 0].legend(loc="upper right")
        axs[1, 0].grid(True, alpha=0.3)
        ticks2 = axs[1, 0].get_xticks()
        axs[1, 0].set_xticks(ticks2)
        axs[1, 0].set_xticklabels(['${:,.0f}'.format(x) for x in ticks2], rotation=30)
        
        # --- Plot 4 (Bottom Right): Simulated Daily Moves Dispersion ---
        daily_ret_jump = np.diff(jump_sims, axis=0).flatten() * 100
        daily_ret_shock = np.diff(shock_sims, axis=0).flatten() * 100
        
        axs[1, 1].hist(daily_ret_jump, bins=50, alpha=0.5, color='blue', density=True, label='Normal Volatility')
        axs[1, 1].hist(daily_ret_shock, bins=50, alpha=0.5, color='red', density=True, label='Shock Volatility')
        axs[1, 1].set_title("4. Dispersion of Daily Simulated Moves (%)")
        axs[1, 1].set_xlabel("Daily Move (% of Initial Portfolio)")
        axs[1, 1].set_ylabel("Density")
        axs[1, 1].legend(loc="upper right")
        axs[1, 1].grid(True, alpha=0.3)

        # Final spacing and save
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.savefig("cli_shock_simulation.png", dpi=300)
        print(f"{Colors.OKGREEN}Success! Advanced 4-panel graph saved as 'cli_shock_simulation.png'.{Colors.ENDC}\n")
    except Exception as e:
        print(f"{Colors.FAIL}Notice: Could not generate graph ({e}).{Colors.ENDC}\n")

if __name__ == "__main__":
    try:
        simulate()
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Simulation aborted by user.{Colors.ENDC}")
        sys.exit(0)
