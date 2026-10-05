"""
Stock Performance Comparison Dashboard with Technical Analysis Strategies.
Author: Dr.-Ing. Diyar Altinses
Version: 3.3.1 (mit korrigierter JS-Escaping-Syntax)
License: MIT
"""

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json

# =============================================================================
# Configuration
# =============================================================================

TICKERS = ["RHM.DE", "NVDA", "GOOGL", "META", "MSFT", "1810.HK", "TSM",
           "AMZN", "AAPL", "BABA", "TSLA", "ASML", "INTC", "AMD",
           "PLTR", "AVGO"]

TICKER_NAMES = {
    "RHM.DE": "Rheinmetall AG", "NVDA": "NVIDIA Corporation",
    "GOOGL": "Alphabet Inc.", "META": "Meta Platforms Inc.",
    "MSFT": "Microsoft", "1810.HK": "Xiaomi Corporation",
    "TSM": "TSMC", "AMZN": "Amazon.com Inc.", "AAPL": "Apple Inc.",
    "BABA": "Alibaba Group", "TSLA": "Tesla Inc.", "ASML": "ASML Holding N.V.",
    "INTC": "Intel Corporation", "AMD": "AMD Inc.", "PLTR": "Palantir Technologies",
    "AVGO": "Broadcom Inc."
}

TIMEFRAMES = {
    "1 Monat": 30,
    "3 Monate": 91,
    "6 Monate": 182,
    "9 Monate": 273,
    "1 Jahr": 365,
    "2 Jahre": 730,
    "3 Jahre": 1095
}

START_CAPITAL = 10000.0

def get_state_color(state):
    if state == "LONG":
        return '#d1fae5'
    elif state == "SHORT":
        return '#fee2e2'
    else:
        return '#f9fafb'

# =============================================================================
# Data Processing Functions
# =============================================================================

def process_full_history(ticker):
    """Lädt die Maximalhistorie und berechnet Indikatoren vor."""
    today = datetime.now()
    start_date = today - timedelta(days=1200)
    
    df = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), 
                     end=today.strftime('%Y-%m-%d'), interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    if df.empty or len(df) < 50:
        return None
    
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['Signal_Strength_Pct'] = 100 - ((df['RSI'] - 30) * (100 / 20))
    df['Signal_Strength_Pct'] = df['Signal_Strength_Pct'].clip(lower=0, upper=100)
    
    score_rsi = df['Signal_Strength_Pct']
    sma_dist_pct = (df['Close'] - df['SMA_20']) / df['SMA_20'] * 100
    score_trend = (-sma_dist_pct * 20).clip(lower=0, upper=100)
    df['Fuzzy_Total_Score'] = (score_rsi * 0.7) + (score_trend * 0.3)
    df['Fuzzy_Total_Score'] = df['Fuzzy_Total_Score'].clip(lower=0, upper=100)
    
    return df

def calculate_strategy_for_slice(df_slice):
    if df_slice.empty:
        return None
        
    cash, shares = START_CAPITAL, 0.0
    portfolio_values = []
    buy_signals = np.full(len(df_slice), np.nan)
    sell_signals = np.full(len(df_slice), np.nan)
    for i in range(len(df_slice)):
        price, rsi = df_slice['Close'].iloc[i], df_slice['RSI'].iloc[i]
        if pd.notna(rsi) and pd.notna(price):
            if rsi <= 30 and cash > 0:
                shares, cash = cash / price, 0.0
                buy_signals[i] = price
            elif rsi >= 50 and shares > 0:
                cash, shares = shares * price, 0.0
                sell_signals[i] = price
        portfolio_values.append(cash + (shares * price if pd.notna(price) else 0))
    df_slice['Portfolio_Value'] = portfolio_values
    
    fuzzy_cash, fuzzy_shares = START_CAPITAL, 0.0
    fuzzy_portfolio_values = []
    fuzzy_buy_signals = np.full(len(df_slice), np.nan)
    fuzzy_sell_signals = np.full(len(df_slice), np.nan)
    for i in range(len(df_slice)):
        price, f_score = df_slice['Close'].iloc[i], df_slice['Fuzzy_Total_Score'].iloc[i]
        if pd.notna(f_score) and pd.notna(price):
            if f_score >= 80 and fuzzy_cash > 0:
                fuzzy_shares, fuzzy_cash = fuzzy_cash / price, 0.0
                fuzzy_buy_signals[i] = price
            elif f_score <= 20 and fuzzy_shares > 0:
                fuzzy_cash, fuzzy_shares = fuzzy_shares * price, 0.0
                fuzzy_sell_signals[i] = price
        fuzzy_portfolio_values.append(fuzzy_cash + (fuzzy_shares * price if pd.notna(price) else 0))
    df_slice['Fuzzy_Portfolio_Value'] = fuzzy_portfolio_values
    
    end_value = df_slice['Portfolio_Value'].iloc[-1]
    profit_pct = (end_value / START_CAPITAL - 1) * 100
    fuzzy_end_value = df_slice['Fuzzy_Portfolio_Value'].iloc[-1]
    fuzzy_profit_pct = (fuzzy_end_value / START_CAPITAL - 1) * 100
    valid_close = df_slice['Close'].dropna()
    buy_and_hold_pct = (valid_close.iloc[-1] / valid_close.iloc[0] - 1) * 100 if len(valid_close) > 0 else 0.0
    
    current_rsi = df_slice['RSI'].iloc[-1] if pd.notna(df_slice['RSI'].iloc[-1]) else 50.0
    current_fuzzy_score = df_slice['Fuzzy_Total_Score'].iloc[-1] if pd.notna(df_slice['Fuzzy_Total_Score'].iloc[-1]) else 50.0
    
    rsi_state = "LONG" if current_rsi <= 30 else ("SHORT" if current_rsi >= 50 else "NEUTRAL")
    fuzzy_state = "LONG" if current_fuzzy_score >= 80 else ("SHORT" if current_fuzzy_score <= 20 else "NEUTRAL")
    
    rsi_long_safety = 100.0 if current_rsi <= 30 else (0.0 if current_rsi >= 50 else (50 - current_rsi) / 20 * 100)
    rsi_short_safety = 100.0 if current_rsi >= 50 else (0.0 if current_rsi <= 30 else (current_rsi - 30) / 20 * 100)
    fuzzy_long_safety = 100.0 if current_fuzzy_score >= 80 else (0.0 if current_fuzzy_score <= 20 else (current_fuzzy_score - 20) / 60 * 100)
    fuzzy_short_safety = 100.0 if current_fuzzy_score <= 20 else (0.0 if current_fuzzy_score >= 80 else (80 - current_fuzzy_score) / 60 * 100)
    
    rsi_display_safety = rsi_long_safety if rsi_state != "SHORT" else rsi_short_safety
    fuzzy_display_safety = fuzzy_long_safety if fuzzy_state != "SHORT" else fuzzy_short_safety
    
    return {
        'df': df_slice,
        'buy_signals': buy_signals, 'sell_signals': sell_signals,
        'fuzzy_buy_signals': fuzzy_buy_signals, 'fuzzy_sell_signals': fuzzy_sell_signals,
        'profit_pct': profit_pct, 'fuzzy_profit_pct': fuzzy_profit_pct,
        'buy_and_hold_pct': buy_and_hold_pct,
        'end_value': end_value, 'fuzzy_end_value': fuzzy_end_value,
        'profit_eur': end_value - START_CAPITAL, 'fuzzy_profit_eur': fuzzy_end_value - START_CAPITAL,
        'rsi_state': rsi_state, 'rsi_score': current_rsi, 'rsi_safety': rsi_display_safety,
        'fuzzy_state': fuzzy_state, 'fuzzy_score': current_fuzzy_score, 'fuzzy_safety': fuzzy_display_safety,
        'start_date': valid_close.index[0].strftime('%Y-%m-%d'),
        'end_date': valid_close.index[-1].strftime('%Y-%m-%d')
    }

# =============================================================================
# Main Execution & Data Preparation
# =============================================================================

print("Lade Daten und berechne Zeiträume...")
master_data = {}
for ticker in TICKERS:
    full_df = process_full_history(ticker)
    if full_df is not None:
        master_data[ticker] = {}
        for tf_name, tf_days in TIMEFRAMES.items():
            cutoff_date = datetime.now() - timedelta(days=tf_days)
            df_slice = full_df[full_df.index >= cutoff_date].copy()
            if len(df_slice) > 10:
                master_data[ticker][tf_name] = calculate_strategy_for_slice(df_slice)

valid_tickers = [t for t in TICKERS if t in master_data and len(master_data[t]) > 0]
if not valid_tickers:
    print("Keine Daten verfügbar.")
    exit(1)

# Create figure
fig = make_subplots(
    rows=5, cols=1, shared_xaxes=False, vertical_spacing=0.02,
    row_heights=[0.18, 0.25, 0.13, 0.13, 0.37],
    specs=[[{"type": "table"}], [{"type": "xy"}], [{"type": "xy"}], [{"type": "xy"}], [{"type": "table"}]],
    subplot_titles=("Performance Evaluation", "Stock Price with Fuzzy Signals", "RSI (14) Indicator", "Buy Signal Strength %", "All Tickers Summary")
)

combination_mapping = []
traces_per_combination = 10
trace_counter = 0

first_ticker = valid_tickers[0]
first_tf = "1 Jahr" if "1 Jahr" in TIMEFRAMES else list(TIMEFRAMES.keys())[2]

for ticker in valid_tickers:
    for tf_name in TIMEFRAMES.keys():
        if tf_name not in master_data[ticker]:
            continue
            
        td = master_data[ticker][tf_name]
        df = td['df']
        is_visible = (ticker == first_ticker and tf_name == first_tf)
        
        # 1. Performance Table
        fig.add_trace(go.Table(
            header=dict(values=[f"<b>Metric ({TICKER_NAMES.get(ticker, ticker)} - {tf_name})</b>", "<b>Result</b>"], fill_color='#e2e8f0', font=dict(size=14)),
            cells=dict(values=[
                ["Start Date", "End Date", "Start Capital", "End (RSI)", "Profit EUR [RSI]", "Return RSI", "End (Fuzzy)", "Profit EUR [Fuzzy]", "Return Fuzzy", "Buy&Hold"],
                [td['start_date'], td['end_date'], f"{START_CAPITAL:,.2f}", f"{td['end_value']:,.2f}", f"{td['profit_eur']:+,.2f}", f"<b>{td['profit_pct']:+.2f}%</b>",
                 f"{td['fuzzy_end_value']:,.2f}", f"{td['fuzzy_profit_eur']:+,.2f}", f"<b>{td['fuzzy_profit_pct']:+.2f}%</b>", f"{td['buy_and_hold_pct']:+.2f}%"]
            ], fill_color='white', font=dict(size=11), height=24),
            visible=is_visible
        ), row=1, col=1)
        
        # 2. Price chart
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Price',
                                   line=dict(color='#1A365D', width=2), showlegend=True, visible=is_visible), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], mode='lines', name='SMA 20',
                                   line=dict(color='#F59E0B', dash='dot', width=1.5), showlegend=True, visible=is_visible), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode='lines', name='SMA 50',
                                   line=dict(color='#EF4444', dash='longdash', width=1.5), showlegend=True, visible=is_visible), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=td['fuzzy_buy_signals'], mode='markers', name='Fuzzy Buy',
                                   marker=dict(color='#059669', size=10, symbol='hexagram'), visible=is_visible), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=td['fuzzy_sell_signals'], mode='markers', name='Fuzzy Sell',
                                   marker=dict(color='#DC2626', size=10, symbol='x'), visible=is_visible), row=2, col=1)
        
        # 3. RSI plot
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI',
                                   line=dict(color='#8B5CF6'), showlegend=True, visible=is_visible), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=[30]*len(df), mode='lines', name='RSI 30 (Buy)',
                                   line=dict(color='green', dash='dash', width=1), showlegend=True, visible=is_visible), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=[50]*len(df), mode='lines', name='RSI 50 (Sell)',
                                   line=dict(color='red', dash='dash', width=1), showlegend=True, visible=is_visible), row=3, col=1)
        
        # 4. Signal Strength
        fig.add_trace(go.Scatter(x=df.index, y=df['Signal_Strength_Pct'], mode='lines', name='Signal Strength %',
                                   line=dict(color='#10B981', width=2), fill='tozeroy', showlegend=True, visible=is_visible), row=4, col=1)
        
        combination_mapping.append({
            'ticker': ticker,
            'tf': tf_name,
            'start': trace_counter,
            'count': traces_per_combination
        })
        trace_counter += traces_per_combination

# Summary table (Row 5)
summary_data_list = []
for ticker in valid_tickers:
    eval_tf = "1 Jahr" if "1 Jahr" in master_data[ticker] else list(master_data[ticker].keys())[0]
    d = master_data[ticker][eval_tf]
    summary_data_list.append({
        'ticker': ticker, 'company_name': TICKER_NAMES.get(ticker, "Unknown"),
        'rsi_state': d['rsi_state'], 'rsi_score': d['rsi_score'], 'rsi_safety': d['rsi_safety'],
        'fuzzy_state': d['fuzzy_state'], 'fuzzy_score': d['fuzzy_score'], 'fuzzy_safety': d['fuzzy_safety']
    })

fig.add_trace(go.Table(
    header=dict(
        values=["Ticker", "Company", "RSI State", "RSI Score", "RSI Safety %", "Fuzzy State", "Fuzzy Score", "Fuzzy Safety %"],
        fill_color='#1e293b', align='center', font=dict(color='white', size=10)
    ),
    cells=dict(
        values=[
            [d['ticker'] for d in summary_data_list],
            [d['company_name'] for d in summary_data_list],
            [d['rsi_state'] for d in summary_data_list],
            [f"{d['rsi_score']:.1f}" for d in summary_data_list],
            [f"{d['rsi_safety']:.1f}%" for d in summary_data_list],
            [d['fuzzy_state'] for d in summary_data_list],
            [f"{d['fuzzy_score']:.1f}" for d in summary_data_list],
            [f"{d['fuzzy_safety']:.1f}%" for d in summary_data_list]
        ],
        fill_color=[
            [get_state_color(d['rsi_state']) for d in summary_data_list],
            ['#ffffff']*len(summary_data_list),
            [get_state_color(d['rsi_state']) for d in summary_data_list],
            ['#ffffff']*len(summary_data_list), ['#ffffff']*len(summary_data_list),
            [get_state_color(d['fuzzy_state']) for d in summary_data_list],
            ['#ffffff']*len(summary_data_list), ['#ffffff']*len(summary_data_list)
        ],
        align='center', font=dict(color='black', size=9), height=24
    ),
    visible=True
), row=5, col=1)

total_traces = trace_counter + 1

fig.update_layout(
    template="plotly_white", hovermode='x unified', height=2400,
    title=dict(text=f"<b>Stock Performance & Strategy Dashboard</b>", x=0.03, y=0.97, font=dict(size=16)),
    plot_bgcolor='white', paper_bgcolor='white', margin=dict(l=40, r=40, t=100, b=40)
)
fig.update_yaxes(range=[0, 105], row=3, col=1)
fig.update_yaxes(range=[0, 105], row=4, col=1)

# Export and HTML Injection for 2 Independent Dropdowns
html_content = fig.to_html(include_plotlyjs='cdn')
html_content = html_content.replace('<head>', '<head>\n<meta name="viewport" content="width=device-width, initial-scale=1.0">')

# HTML Dropdown Controls & JS Code einfügen (mit doppelt maskierten Klammern für den f-string)
controls_html = f"""
<div style="background: #f8fafc; border: 1px solid #cbd5e1; padding: 12px 20px; border-radius: 8px; margin: 10px 40px; display: flex; gap: 20px; align-items: center; font-family: sans-serif; font-size: 14px;">
    <div>
        <label for="tickerSelect" style="font-weight: bold; margin-right: 8px; color: #1e293b;">Ticker:</label>
        <select id="tickerSelect" style="padding: 6px 12px; border-radius: 6px; border: 1px solid #cbd5e1; background: white; font-size: 14px;">
            {''.join([f'<option value="{t}" {"selected" if t == first_ticker else ""}>{TICKER_NAMES.get(t, t)} ({t})</option>' for t in valid_tickers])}
        </select>
    </div>
    <div>
        <label for="tfSelect" style="font-weight: bold; margin-right: 8px; color: #1e293b;">Zeitraum:</label>
        <select id="tfSelect" style="padding: 6px 12px; border-radius: 6px; border: 1px solid #cbd5e1; background: white; font-size: 14px;">
            {''.join([f'<option value="{tf}" {"selected" if tf == first_tf else ""}>{tf}</option>' for tf in TIMEFRAMES.keys()])}
        </select>
    </div>
</div>

<script>
const comboMap = {json.dumps(combination_mapping)};
const totalTracesCount = {total_traces};

function updateDashboard() {{
    const selTicker = document.getElementById('tickerSelect').value;
    const selTf = document.getElementById('tfSelect').value;
    
    let visibility = new Array(totalTracesCount).fill(false);
    // Summary table bleibt immer sichtbar (letzter Trace)
    visibility[totalTracesCount - 1] = true;
    
    comboMap.forEach(combo => {{
        if (combo.ticker === selTicker && combo.tf === selTf) {{
            for (let i = 0; i < combo.count; i++) {{
                visibility[combo.start + i] = true;
            }}
        }}
    }});
    
    const gd = document.querySelector('.plotly-graph-div');
    if (gd) {{
        Plotly.restyle(gd, {{ visible: visibility }});
    }}
}}

document.getElementById('tickerSelect').addEventListener('change', updateDashboard);
document.getElementById('tfSelect').addEventListener('change', updateDashboard);
</script>
"""

# Füge die Steuerungselemente nach dem Öffnen des Body-Tags ein
html_content = html_content.replace('<body>', f'<body>\n{controls_html}')

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Generierte index.html mit zwei unabhängigen Dropdowns für {len(valid_tickers)} Ticker.")