# Financial Shock Portfolio Simulator

Stress-test your investment portfolio against historical black swan events using Monte Carlo Jump-Diffusion models.

This project allows you to simulate how your active investment portfolio would have historically performed during massive economic tragedies, using real data from the S&P 500, Nasdaq, individual stocks, and more. 

## Features
- **Dynamic Asset Allocation:** Create any portfolio combination using Yahoo Finance tickers (e.g., AAPL, SPY, GC=F).
- **Historical Black Swans:** Simulate against events like the 2008 Financial Crisis, COVID-19 pandemic crash, Dot-Com Bubble, and 9/11.
- **Monte Carlo Generation:** Uses advanced correlated Geometric Brownian Motion with Jump-Diffusion mathematics to compute 500 potential future trajectories.
- **Risk Mitigation Analysis:** Automatically calculates actionable safety strategies, such as precise 10% trailing stop-losses, and Option Hedging cost requirements.
- **Graphical Dashboard:** Automatically generates and saves a highly-detailed 4-panel mathematical dashboard (`cli_shock_simulation.png`) for deep visual analysis.

## Setup & Installation

1. Create a virtual environment (optional but recommended):
```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install the necessary quantitative libraries:
```bash
pip install yfinance pandas numpy matplotlib scipy
```

## How to Run

Launch the interactive CLI simulator:
```bash
python cli.py
```

Follow the prompts to enter your capital, add asset tickers, assign their weights, and select the historical shock you want to test against.
