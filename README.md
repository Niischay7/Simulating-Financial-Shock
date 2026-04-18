# Financial Risk & Portfolio Management Simulator

A dual-mode CLI application that acts as both a **Live Portfolio Diagnostics Tracker** and an advanced **Historical Black Swan Simulator**. 

This system allows traders and investors to track their real-time performance and mathematically stress-test how their current holdings would survive during massive economic tragedies, utilizing Monte Carlo Jump-Diffusion models and Modern Portfolio Theory.

## Features

### Mode 1: Live Portfolio Tracker & Diagnostics
- **Real-Time P&L Tracking:** Input your holdings (shares and average cost) to automatically fetch live market prices and calculate your exact Dollar and Percentage P&L.
- **Quantitative Performance Metrics:** Automatically calculates 1-Year Annualized Returns, Volatility, Sharpe Ratio, and Maximum Drawdown.
- **Health Dashboard:** Generates a 4-panel `portfolio_diagnostics.png` visualization containing historical growth paths, asset allocation pie charts, and normalized drawdown profiles.

### Mode 2: Historical Shock Simulator
- **Historical Black Swans:** Simulate against events like the 2008 Financial Crisis, COVID-19 pandemic crash, Dot-Com Bubble, and 9/11.
- **Modern Portfolio Theory (MPT) Optimization:** Instead of guessing weights, use our built-in engine to automatically allocate your assets for Maximum Sharpe Ratio or Minimum Volatility based on historical correlations.
- **Monte Carlo Generation:** Uses advanced correlated Geometric Brownian Motion with Jump-Diffusion mathematics to compute 500 potential future trajectories.
- **Risk Mitigation Analysis:** Automatically calculates actionable safety strategies, such as precise 10% trailing stop-losses.
- **Graphical Dashboard:** Automatically generates and saves a highly detailed `cli_shock_simulation.png` mathematical dashboard for deep visual analysis.

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

Follow the on-screen prompts to select your mode, build your portfolio, and generate advanced analytics.
