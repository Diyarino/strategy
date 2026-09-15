"""
Stock Performance Comparison Dashboard with Technical Analysis Strategies.

This module provides a comprehensive stock analysis tool that compares multiple
trading strategies (RSI-based and Fuzzy Logic-based) against buy-and-hold
performance across a portfolio of technology stocks.

Author: Dr.-Ing. Diyar Altinses
Version: 1.0.0
License: MIT
"""

# =============================================================================
# Imports
# =============================================================================

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =============================================================================
# Configuration Constants
# =============================================================================

# List of stock tickers available for selection in the dropdown menu
TICKERS = ["RHM.DE", "NVDA", "GOOGL", "META", "MSFT", "1810.HK", "TSM",
           "ISRG", "AMZN", "AAPL", "BABA", "TSLA", "ASML", "INTC", "AMD",
           "ENPH", "PLTR", "CRWD", "AVGO"]

# Initial investment capital for backtesting simulations
START_CAPITAL = 10000.0

# =============================================================================
# Plot Configuration
# =============================================================================

# Create Plotly subplot structure with 5 rows and 2 columns
# Layout specifications:
#   Row 1: Performance metrics table (spans both columns)
#   Row 2: Price charts with strategy signals (split into 2 columns)
#   Row 3: RSI indicator visualization (spans both columns)
#   Row 4: Buy signal strength indicator (spans both columns)
#   Row 5: Fuzzy logic total score (spans both columns)
fig = make_subplots(
    rows=5, cols=2,
    shared_xaxes=False,
    vertical_spacing=0.06,
    horizontal_spacing=0.08,
    row_heights=[0.28, 0.26, 0.15, 0.15, 0.16],
    specs=[
        [{"type": "table", "colspan": 2}, None],
        [{"type": "xy"}, {"type": "xy"}],
        [{"type": "xy", "colspan": 2}, None],
        [{"type": "xy", "colspan": 2}, None],
        [{"type": "xy", "colspan": 2}, None]
    ],
    subplot_titles=(
        "Performance Evaluation (Strategy Comparison)",
        "1. RSI Strategy (Buy <= 30, Sell >= 50)",
        "2. Fuzzy Strategy (Buy >= 80, Sell <= 20)",
        "RSI (14) Indicator",
        "Buy Signal Strength in % (Indicator)",
        "Fuzzy Total Score (Multi-Indicator in %)"
    )
)

# Initialize counters and collections for trace management
all_traces_count = 0
dropdown_buttons = []

# =============================================================================
# Stock Data Processing Loop
# =============================================================================

for idx, ticker in enumerate(TICKERS):
    # -------------------------------------------------------------------------
    # Step 1: Download Historical Price Data (Last 12 Months, Daily Interval)
    # -------------------------------------------------------------------------
    df = yf.download(ticker, period="1y", interval="1d", progress=False)

    # Handle MultiIndex columns from yfinance (common with newer versions)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Skip stocks with insufficient data for meaningful analysis
    if df.empty or len(df) < 50:
        continue

    # -------------------------------------------------------------------------
    # Step 2: Calculate Technical Indicators
    # -------------------------------------------------------------------------

    # Simple Moving Averages for trend analysis
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()

    # Relative Strength Index (RSI) - 14-day period
    # RSI measures the speed and magnitude of price changes
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # -------------------------------------------------------------------------
    # Step 3: Calculate Continuous Signal Strength Percentage
    # -------------------------------------------------------------------------
    # Normalizes RSI values to a 0-100% scale for buy signal strength
    # RSI of 30 corresponds to 100% signal strength, RSI of 50 to 0%
    df['Signal_Strength_Pct'] = 100 - ((df['RSI'] - 30) * (100 / 20))
    df['Signal_Strength_Pct'] = df['Signal_Strength_Pct'].clip(lower=0, upper=100)

    # -------------------------------------------------------------------------
    # Step 4: Fuzzy Logic Multi-Indicator Score Calculation
    # -------------------------------------------------------------------------
    # Combines RSI-based signal strength with trend analysis
    # Weight: 70% RSI, 30% Trend (SMA distance)

    # RSI component (already calculated as signal strength)
    score_rsi = df['Signal_Strength_Pct']

    # Trend component: Distance from 20-day SMA
    # Negative distance (price below SMA) = oversold = higher score
    sma_dist_pct = (df['Close'] - df['SMA_20']) / df['SMA_20'] * 100
    score_trend = (-sma_dist_pct * 20).clip(lower=0, upper=100)

    # Weighted combination of both indicators
    df['Fuzzy_Total_Score'] = (score_rsi * 0.7) + (score_trend * 0.3)
    df['Fuzzy_Total_Score'] = df['Fuzzy_Total_Score'].clip(lower=0, upper=100)

    # -------------------------------------------------------------------------
    # Step 5: RSI Strategy Backtest (Buy <= 30, Sell >= 50)
    # -------------------------------------------------------------------------
    cash = START_CAPITAL
    shares = 0.0
    portfolio_values = []
    buy_signals = np.full(len(df), np.nan)
    sell_signals = np.full(len(df), np.nan)

    for i in range(len(df)):
        price = df['Close'].iloc[i]
        rsi = df['RSI'].iloc[i]

        if pd.notna(rsi) and pd.notna(price):
            # Buy signal: RSI <= 30 (oversold condition)
            if rsi <= 30 and cash > 0:
                shares = cash / price
                cash = 0.0
                buy_signals[i] = price
            # Sell signal: RSI >= 50 (recovery achieved)
            elif rsi >= 50 and shares > 0:
                cash = shares * price
                shares = 0.0
                sell_signals[i] = price

        # Calculate total portfolio value at each time step
        portfolio_values.append(cash + (shares * price if pd.notna(price) else 0))

    df['Portfolio_Value'] = portfolio_values

    # -------------------------------------------------------------------------
    # Step 6: Fuzzy Strategy Backtest (Buy >= 80, Sell <= 20)
    # -------------------------------------------------------------------------
    fuzzy_cash = START_CAPITAL
    fuzzy_shares = 0.0
    fuzzy_portfolio_values = []
    fuzzy_buy_signals = np.full(len(df), np.nan)
    fuzzy_sell_signals = np.full(len(df), np.nan)

    for i in range(len(df)):
        price = df['Close'].iloc[i]
        f_score = df['Fuzzy_Total_Score'].iloc[i]

        if pd.notna(f_score) and pd.notna(price):
            # Buy signal: Fuzzy Score >= 80 (strong buy signal)
            if f_score >= 80 and fuzzy_cash > 0:
                fuzzy_shares = fuzzy_cash / price
                fuzzy_cash = 0.0
                fuzzy_buy_signals[i] = price
            # Sell signal: Fuzzy Score <= 20 (strong sell signal)
            elif f_score <= 20 and fuzzy_shares > 0:
                fuzzy_cash = fuzzy_shares * price
                fuzzy_shares = 0.0
                fuzzy_sell_signals[i] = price

        # Calculate total portfolio value at each time step
        fuzzy_portfolio_values.append(
            fuzzy_cash + (fuzzy_shares * price if pd.notna(price) else 0)
        )

    df['Fuzzy_Portfolio_Value'] = fuzzy_portfolio_values

    # -------------------------------------------------------------------------
    # Step 7: Calculate Performance Metrics
    # -------------------------------------------------------------------------

    # RSI Strategy Performance
    end_value = df['Portfolio_Value'].iloc[-1]
    profit_eur = end_value - START_CAPITAL
    profit_pct = (end_value / START_CAPITAL - 1) * 100

    # Fuzzy Strategy Performance
    fuzzy_end_value = df['Fuzzy_Portfolio_Value'].iloc[-1]
    fuzzy_profit_eur = fuzzy_end_value - START_CAPITAL
    fuzzy_profit_pct = (fuzzy_end_value / START_CAPITAL - 1) * 100

    # Buy & Hold Benchmark
    buy_price = df['Close'].dropna().iloc[0]
    end_price = df['Close'].dropna().iloc[-1]
    buy_and_hold_pct = (end_price / buy_price - 1) * 100

    # Determine visibility: only the first ticker is visible initially
    is_visible = (idx == 0)

    # -------------------------------------------------------------------------
    # Step 8: Add Visualization Traces
    # -------------------------------------------------------------------------

    # Performance metrics table (Row 1, Column 1)
    fig.add_trace(go.Table(
        header=dict(
            values=[f"<b>Metric / Strategy ({ticker})</b>", "<b>Value / Result</b>"],
            fill_color='#e2e8f0',
            align='left',
            font=dict(color='black', size=14)
        ),
        cells=dict(
            values=[
                [
                    "Start Capital",
                    "End Capital (RSI 30/50)",
                    "Profit/Loss (EUR) [RSI]",
                    "Return RSI Strategy",
                    "End Capital (Fuzzy >=80/<=20)",
                    "Profit/Loss (EUR) [Fuzzy]",
                    "Return Fuzzy Strategy",
                    "Buy & Hold Benchmark Return"
                ],
                [
                    f"{START_CAPITAL:,.2f} EUR",
                    f"{end_value:,.2f} EUR",
                    f"{profit_eur:+,.2f} EUR",
                    f"<b>{profit_pct:+.2f} %</b>",
                    f"{fuzzy_end_value:,.2f} EUR",
                    f"{fuzzy_profit_eur:+,.2f} EUR",
                    f"<b>{fuzzy_profit_pct:+.2f} %</b>",
                    f"{buy_and_hold_pct:+.2f} %"
                ]
            ],
            fill_color='white',
            align='left',
            font=dict(
                color=[
                    'black', 'black', 'black',
                    'green' if profit_pct > 0 else 'red',
                    'black', 'black',
                    'green' if fuzzy_profit_pct > 0 else 'red',
                    'black'
                ],
                size=13
            ),
            height=30
        )
    ), row=1, col=1)

    # RSI Strategy Chart (Row 2, Column 1)
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Close'],
            mode='lines',
            name='Price',
            line=dict(color='#1A365D', width=2),
            showlegend=False,
            visible=is_visible
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['SMA_20'],
            mode='lines',
            name='SMA 20',
            line=dict(color='#F59E0B', dash='dot'),
            showlegend=False,
            visible=is_visible
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=buy_signals,
            mode='markers',
            name='RSI Buy',
            marker=dict(
                color='green',
                size=12,
                symbol='triangle-up',
                line=dict(color='darkgreen', width=1)
            ),
            visible=is_visible
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=sell_signals,
            mode='markers',
            name='RSI Sell',
            marker=dict(
                color='red',
                size=12,
                symbol='triangle-down',
                line=dict(color='darkred', width=1)
            ),
            visible=is_visible
        ),
        row=2, col=1
    )

    # Fuzzy Strategy Chart (Row 2, Column 2)
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Close'],
            mode='lines',
            name='Price',
            line=dict(color='#64748B', width=1.5),
            showlegend=False,
            visible=is_visible
        ),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=fuzzy_buy_signals,
            mode='markers',
            name='Fuzzy Buy',
            marker=dict(
                color='#059669',
                size=12,
                symbol='hexagram',
                line=dict(color='black', width=1)
            ),
            visible=is_visible
        ),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=fuzzy_sell_signals,
            mode='markers',
            name='Fuzzy Sell',
            marker=dict(
                color='#DC2626',
                size=12,
                symbol='x',
                line=dict(color='black', width=1)
            ),
            visible=is_visible
        ),
        row=2, col=2
    )

    # RSI Indicator Chart (Row 3, Column 1)
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['RSI'],
            mode='lines',
            name='RSI',
            line=dict(color='#8B5CF6'),
            showlegend=False,
            visible=is_visible
        ),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=[30] * len(df),
            mode='lines',
            name='30 Threshold',
            line=dict(color='green', dash='dash', width=1.5),
            showlegend=False,
            opacity=0.7,
            visible=is_visible
        ),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=[50] * len(df),
            mode='lines',
            name='50 Threshold',
            line=dict(color='red', dash='dash', width=1.5),
            showlegend=False,
            opacity=0.7,
            visible=is_visible
        ),
        row=3, col=1
    )

    # Buy Signal Strength Chart (Row 4, Column 1)
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Signal_Strength_Pct'],
            mode='lines',
            name='Buy Signal (%)',
            line=dict(color='#10B981', width=2),
            fill='tozeroy',
            fillcolor='rgba(16, 185, 129, 0.15)',
            showlegend=False,
            visible=is_visible
        ),
        row=4, col=1
    )

    # Fuzzy Total Score Chart (Row 5, Column 1)
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Fuzzy_Total_Score'],
            mode='lines',
            name='Fuzzy Score (%)',
            line=dict(color='#2563EB', width=2),
            fill='tozeroy',
            fillcolor='rgba(37, 99, 235, 0.15)',
            showlegend=False,
            visible=is_visible
        ),
        row=5, col=1
    )

# =============================================================================
# Dropdown Menu Configuration
# =============================================================================

# Each ticker adds 13 traces to the figure
traces_per_ticker = 13

for i, ticker in enumerate(TICKERS):
    # Create visibility array: all traces hidden except current ticker's
    visibility = [False] * (len(TICKERS) * traces_per_ticker)
    for j in range(traces_per_ticker):
        visibility[i * traces_per_ticker + j] = True

    # Define dropdown button for this ticker
    dropdown_buttons.append({
        'args': [{'visible': visibility}],
        'label': ticker,
        'method': 'update'
    })

# =============================================================================
# Figure Layout Configuration
# =============================================================================

fig.update_layout(
    template="plotly_white",
    hovermode='x unified',
    height=1750,
    title=dict(
        text="<b>RHM & Tech Stocks: Strategy Comparison</b>",
        x=0.03,
        y=0.97,
        font=dict(size=18, color='#1A365D')
    ),
    updatemenus=[{
        'buttons': dropdown_buttons,
        'direction': 'down',
        'showactive': True,
        'active': 0,
        'x': 0.98,
        'y': 1.02,
        'xanchor': 'right',
        'yanchor': 'bottom',
        'bgcolor': '#f8fafc',
        'bordercolor': '#cbd5e1',
        'font': dict(size=13, color='#1e293b')
    }],
    plot_bgcolor='white',
    paper_bgcolor='white',
    margin=dict(l=40, r=40, t=140, b=40)
)

# Fix Y-axis ranges for indicator charts to maintain consistent scale
fig.update_yaxes(range=[0, 105], row=4, col=1)
fig.update_yaxes(range=[0, 105], row=5, col=1)

# =============================================================================
# HTML Export
# =============================================================================

# Generate HTML representation using CDN for Plotly.js
html_content = fig.to_html(include_plotlyjs='cdn')

# Inject PWA meta tags for mobile app compatibility
html_content = html_content.replace(
    '<head>',
    '<head>\n<link rel="manifest" href="manifest.json">\n<meta name="apple-mobile-web-app-capable" content="yes">'
)

# Write final HTML to index.html
with open("index.html", "w", encoding="utf-8") as f:
    f.write(html_content)
