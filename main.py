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
           "AMZN", "AAPL", "BABA", "TSLA", "ASML", "INTC", "AMD",
           "PLTR", "AVGO"]

# Dictionary mapping tickers to company names
TICKER_NAMES = {
    "RHM.DE": "Rheinmetall AG",
    "NVDA": "NVIDIA Corporation",
    "GOOGL": "Alphabet Inc.",
    "META": "Meta Platforms Inc.",
    "MSFT": "Microsoft",
    "1810.HK": "Xiaomi Corporation",
    "TSM": "TSMC",
    "AMZN": "Amazon.com Inc.",
    "AAPL": "Apple Inc.",
    "BABA": "Alibaba Group",
    "TSLA": "Tesla Inc.",
    "ASML": "ASML Holding N.V.",
    "INTC": "Intel Corporation",
    "AMD": "AMD Inc.",
    "PLTR": "Palantir Technologies",
    "AVGO": "Broadcom Inc."
}

# Initial investment capital for backtesting simulations
START_CAPITAL = 10000.0

# =============================================================================
# Plot Configuration
# =============================================================================

# Create Plotly subplot structure with 6 rows and 2 columns
# Layout specifications:
#   Row 1: Performance metrics table (spans both columns)
#   Row 2: Price charts with strategy signals (split into 2 columns)
#   Row 3: RSI indicator visualization (spans both columns)
#   Row 4: Buy signal strength indicator (spans both columns)
#   Row 5: Fuzzy logic total score (spans both columns)
#   Row 6: Summary table with ticker info and safety percentages (spans both columns)
fig = make_subplots(
    rows=6, cols=2,
    shared_xaxes=False,
    vertical_spacing=0.02,
    horizontal_spacing=0.08,
    row_heights=[0.16, 0.15, 0.11, 0.11, 0.11, 0.36],
    specs=[
        [{"type": "table", "colspan": 2}, None],
        [{"type": "xy"}, {"type": "xy"}],
        [{"type": "xy", "colspan": 2}, None],
        [{"type": "xy", "colspan": 2}, None],
        [{"type": "xy", "colspan": 2}, None],
        [{"type": "table", "colspan": 2}, None]
    ],
    subplot_titles=(
        "Performance Evaluation (Strategy Comparison)",
        "1. RSI Strategy (Buy <= 30, Sell >= 50)",
        "2. Fuzzy Strategy (Buy >= 80, Sell <= 20)",
        "RSI (14) Indicator",
        "Buy Signal Strength in % (Indicator)",
        "Fuzzy Total Score (Multi-Indicator in %)",
        "Summary: Ticker Analysis with Safety Percentages"
    )
)

# Initialize counters and collections for trace management
all_traces_count = 0
dropdown_buttons = []

# Collection for summary table data
summary_data = []

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
    # Calculate Current State and Safety Percentages
    # -------------------------------------------------------------------------
    
    # Get the latest values
    current_rsi = df['RSI'].iloc[-1] if pd.notna(df['RSI'].iloc[-1]) else 50.0
    current_fuzzy_score = df['Fuzzy_Total_Score'].iloc[-1] if pd.notna(df['Fuzzy_Total_Score'].iloc[-1]) else 50.0
    current_price = df['Close'].iloc[-1]
    
    # Determine current state for RSI (Long if RSI <= 30, Short if RSI >= 50, Neutral otherwise)
    if current_rsi <= 30:
        rsi_state = "LONG"
    elif current_rsi >= 50:
        rsi_state = "SHORT"
    else:
        rsi_state = "NEUTRAL"
    
    # Determine current state for Fuzzy (Long if score >= 80, Short if score <= 20, Neutral otherwise)
    if current_fuzzy_score >= 80:
        fuzzy_state = "LONG"
    elif current_fuzzy_score <= 20:
        fuzzy_state = "SHORT"
    else:
        fuzzy_state = "NEUTRAL"
    
    # Calculate safety percentages using linear interpolation
    # RSI Rules: Buy <= 30 (100% safe), Sell >= 50 (0% safe for long)
    # For LONG position safety:
    #   - RSI of 30 = 100% safe (strong buy)
    #   - RSI of 50 = 0% safe (strong sell for long)
    #   - Linear interpolation between 30 and 50
    if current_rsi <= 30:
        rsi_long_safety_pct = 100.0
    elif current_rsi >= 50:
        rsi_long_safety_pct = 0.0
    else:
        # Linear interpolation: (50 - RSI) / (50 - 30) * 100
        rsi_long_safety_pct = (50 - current_rsi) / 20 * 100
    
    # For SHORT position safety (inverse):
    #   - RSI of 50 = 100% safe for short
    #   - RSI of 30 = 0% safe for short
    if current_rsi >= 50:
        rsi_short_safety_pct = 100.0
    elif current_rsi <= 30:
        rsi_short_safety_pct = 0.0
    else:
        rsi_short_safety_pct = (current_rsi - 30) / 20 * 100
    
    # Fuzzy Rules: Buy >= 80 (100% safe for long), Sell <= 20 (0% safe for long)
    # For LONG position safety:
    #   - Score of 80 = 100% safe (strong buy)
    #   - Score of 20 = 0% safe (strong sell)
    #   - Linear interpolation between 20 and 80
    if current_fuzzy_score >= 80:
        fuzzy_long_safety_pct = 100.0
    elif current_fuzzy_score <= 20:
        fuzzy_long_safety_pct = 0.0
    else:
        # Linear interpolation: (Score - 20) / (80 - 20) * 100
        fuzzy_long_safety_pct = (current_fuzzy_score - 20) / 60 * 100
    
    # For SHORT position safety (inverse):
    #   - Score of 20 = 100% safe for short
    #   - Score of 80 = 0% safe for short
    if current_fuzzy_score <= 20:
        fuzzy_short_safety_pct = 100.0
    elif current_fuzzy_score >= 80:
        fuzzy_short_safety_pct = 0.0
    else:
        fuzzy_short_safety_pct = (80 - current_fuzzy_score) / 60 * 100
    
    # Determine which safety percentage to show based on current state
    if rsi_state == "LONG":
        rsi_display_safety = rsi_long_safety_pct
    elif rsi_state == "SHORT":
        rsi_display_safety = rsi_short_safety_pct
    else:
        rsi_display_safety = rsi_long_safety_pct  # Show long safety for neutral
    
    if fuzzy_state == "LONG":
        fuzzy_display_safety = fuzzy_long_safety_pct
    elif fuzzy_state == "SHORT":
        fuzzy_display_safety = fuzzy_short_safety_pct
    else:
        fuzzy_display_safety = fuzzy_long_safety_pct  # Show long safety for neutral
    
    # Store data for summary table
    summary_data.append({
        'ticker': ticker,
        'company_name': TICKER_NAMES.get(ticker, "Unknown"),
        'rsi_state': rsi_state,
        'rsi_score': current_rsi,
        'rsi_safety': rsi_display_safety,
        'fuzzy_state': fuzzy_state,
        'fuzzy_score': current_fuzzy_score,
        'fuzzy_safety': fuzzy_display_safety
    })
    
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
# Create Summary Table with ALL Tickers (after the loop)
# =============================================================================

# Helper function to get background color based on state
def get_state_color(state):
    """Return hex color based on state: LONG=green, SHORT=red, NEUTRAL=white"""
    if state == "LONG":
        return '#d1fae5'  # Light green
    elif state == "SHORT":
        return '#fee2e2'  # Light red
    else:
        return '#f9fafb'  # Light gray/white

# Build column values and colors for the summary table
ticker_col = []
company_col = []
rsi_state_col = []
rsi_score_col = []
rsi_safety_col = []
fuzzy_state_col = []
fuzzy_score_col = []
fuzzy_safety_col = []

rsi_state_colors = []
fuzzy_state_colors = []

for data in summary_data:
    ticker_col.append(data['ticker'])
    company_col.append(data['company_name'])
    rsi_state_col.append(data['rsi_state'])
    rsi_score_col.append(f"{data['rsi_score']:.1f}")
    rsi_safety_col.append(f"{data['rsi_safety']:.1f}%")
    fuzzy_state_col.append(data['fuzzy_state'])
    fuzzy_score_col.append(f"{data['fuzzy_score']:.1f}")
    fuzzy_safety_col.append(f"{data['fuzzy_safety']:.1f}%")
    
    rsi_state_colors.append(get_state_color(data['rsi_state']))
    fuzzy_state_colors.append(get_state_color(data['fuzzy_state']))

# Add the summary table (only visible for first ticker selection, spans all)
fig.add_trace(go.Table(
    header=dict(
        values=[
            "<b>Ticker</b>",
            "<b>Company Name</b>",
            "<b>RSI State</b>",
            "<b>RSI Score</b>",
            "<b>RSI Safety %</b>",
            "<b>Fuzzy State</b>",
            "<b>Fuzzy Score</b>",
            "<b>Fuzzy Safety %</b>"
        ],
        fill_color='#1e293b',
        align='center',
        font=dict(color='white', size=11)
    ),
    cells=dict(
        values=[
            ticker_col,
            company_col,
            rsi_state_col,
            rsi_score_col,
            rsi_safety_col,
            fuzzy_state_col,
            fuzzy_score_col,
            fuzzy_safety_col
        ],
        fill_color=[rsi_state_colors, ['#ffffff'] * len(summary_data), rsi_state_colors,
                    ['#ffffff'] * len(summary_data), ['#ffffff'] * len(summary_data),
                    fuzzy_state_colors, ['#ffffff'] * len(summary_data), ['#ffffff'] * len(summary_data)],
        align='center',
        font=dict(color='black', size=9),
        height=24
    ),
    visible=True  # Always visible
), row=6, col=1)

# =============================================================================
# Dropdown Menu Configuration
# =============================================================================

# Each ticker adds 13 traces to the figure (summary table is always visible)
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
    height=2600,
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
