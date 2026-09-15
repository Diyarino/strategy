import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. Kursdaten laden (Letzte 12 Monate)
df = yf.download("RHM.DE", period="1y", interval="1d")

# # Nach dem Laden der Daten (z.B. mit period="2y"):
# start_date = "2024-09-15"
# end_date = "2025-09-15"

# # DataFrame auf diesen Zeitraum einschränken
# df = df.loc[start_date:end_date]

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# 2. Metriken berechnen
df['SMA_20'] = df['Close'].rolling(window=20).mean()
df['SMA_50'] = df['Close'].rolling(window=50).mean()

# RSI (14 Tage)
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# 3. Kontinuierliche Signalstärke berechnen
df['Signalstaerke_Pct'] = 100 - ((df['RSI'] - 30) * (100 / 20))
df['Signalstaerke_Pct'] = df['Signalstaerke_Pct'].clip(lower=0, upper=100)

# --- Fuzzy-Logic Multi-Indikator Score ---
score_rsi = df['Signalstaerke_Pct']
sma_dist_pct = (df['Close'] - df['SMA_20']) / df['SMA_20'] * 100
score_trend = (-sma_dist_pct * 20).clip(lower=0, upper=100)
df['Fuzzy_Gesamtscore'] = (score_rsi * 0.7) + (score_trend * 0.3)
df['Fuzzy_Gesamtscore'] = df['Fuzzy_Gesamtscore'].clip(lower=0, upper=100)
# ----------------------------------------------------

# 4. Backtest Logik 1 (RSI: Buy <=30, Sell >=50)
start_capital = 10000.0
cash = start_capital
shares = 0.0
portfolio_values = []
buy_signals = np.full(len(df), np.nan)
sell_signals = np.full(len(df), np.nan)

for i in range(len(df)):
    price = df['Close'].iloc[i]
    rsi = df['RSI'].iloc[i]
    if pd.notna(rsi) and pd.notna(price):
        if rsi <= 30 and cash > 0:
            shares = cash / price
            cash = 0.0
            buy_signals[i] = price
        elif rsi >= 50 and shares > 0:
            cash = shares * price
            shares = 0.0
            sell_signals[i] = price
    portfolio_values.append(cash + (shares * price if pd.notna(price) else 0))

df['Portfolio_Value'] = portfolio_values

# --- Backtest Logik 2 (Fuzzy Score Buy >= 80, Sell <= 20) ---
fuzzy_cash = start_capital
fuzzy_shares = 0.0
fuzzy_portfolio_values = []
fuzzy_buy_signals = np.full(len(df), np.nan)
fuzzy_sell_signals = np.full(len(df), np.nan)

for i in range(len(df)):
    price = df['Close'].iloc[i]
    f_score = df['Fuzzy_Gesamtscore'].iloc[i]
    if pd.notna(f_score) and pd.notna(price):
        if f_score >= 80 and fuzzy_cash > 0:
            fuzzy_shares = fuzzy_cash / price
            fuzzy_cash = 0.0
            fuzzy_buy_signals[i] = price
        elif f_score <= 20 and fuzzy_shares > 0:
            fuzzy_cash = fuzzy_shares * price
            fuzzy_shares = 0.0
            fuzzy_sell_signals[i] = price
    fuzzy_portfolio_values.append(fuzzy_cash + (fuzzy_shares * price if pd.notna(price) else 0))

df['Fuzzy_Portfolio_Value'] = fuzzy_portfolio_values
# ----------------------------------------------------------------

# 5. Performance-Metriken berechnen
end_value = df['Portfolio_Value'].iloc[-1]
profit_eur = end_value - start_capital
profit_pct = (end_value / start_capital - 1) * 100

fuzzy_end_value = df['Fuzzy_Portfolio_Value'].iloc[-1]
fuzzy_profit_eur = fuzzy_end_value - start_capital
fuzzy_profit_pct = (fuzzy_end_value / start_capital - 1) * 100

buy_price = df['Close'].dropna().iloc[0]
end_price = df['Close'].dropna().iloc[-1]
buy_and_hold_pct = (end_price / buy_price - 1) * 100

# 6. Plotly Figure erstellen (Grid mit 5 Zeilen, 2 Spalten)
fig = make_subplots(
    rows=5, cols=2, 
    shared_xaxes=False, 
    vertical_spacing=0.03, 
    horizontal_spacing=0.05,
    row_heights=[0.22, 0.34, 0.14, 0.15, 0.15],
    specs=[
        [{"type": "table", "colspan": 2}, None],           # Zeile 1: Große Tabelle über volle Breite
        [{"type": "xy"}, {"type": "xy"}],                  # Zeile 2: Strategie 1 (Links) & Strategie 2 (Rechts) nebeneinander
        [{"type": "xy", "colspan": 2}, None],              # Zeile 3: RSI über volle Breite
        [{"type": "xy", "colspan": 2}, None],              # Zeile 4: Signalstärke über volle Breite
        [{"type": "xy", "colspan": 2}, None]               # Zeile 5: Fuzzy Score über volle Breite
    ],
    subplot_titles=(
        "Performance Auswertung (Strategie-Vergleich)", "",  # Für Zeile 1 (Tabelle)
        "1. RSI-Strategie (Buy <= 30, Sell >= 50)", "2. Fuzzy-Strategie (Buy >= 80, Sell <= 20)",  # Zeile 2 (Nebeneinander)
        "RSI (14) - Indikator", "",                        # Zeile 3
        "Kaufsignal-Stärke in % (Indikator)", "",          # Zeile 4
        "Fuzzy Gesamtscore (Multi-Indikator in %)", ""     # Zeile 5
    )
)

# --- Subplot 1: Große Tabelle (Zeile 1) ---
fig.add_trace(
    go.Table(
        header=dict(
            values=["<b>Metrik / Strategie</b>", "<b>Wert / Ergebnis</b>"],
            fill_color='#e2e8f0', align='left', font=dict(color='black', size=14)
        ),
        cells=dict(
            values=[
                ["Startkapital", "Endkapital (RSI 30/50)", "Gewinn/Verlust (€) [RSI]", "Rendite RSI-Strategie", 
                 "Endkapital (Fuzzy >=80/<=20)", "Gewinn/Verlust (€) [Fuzzy]", "Rendite Fuzzy-Strategie", "Buy & Hold Vergleichs-Rendite"],
                [
                    f"{start_capital:,.2f} €", f"{end_value:,.2f} €", f"{profit_eur:+,.2f} €", f"<b>{profit_pct:+.2f} %</b>",
                    f"{fuzzy_end_value:,.2f} €", f"{fuzzy_profit_eur:+,.2f} €", f"<b>{fuzzy_profit_pct:+.2f} %</b>", f"{buy_and_hold_pct:+.2f} %"
                ]
            ],
            fill_color='white', align='left',
            font=dict(color=['black', 'black', 'black', 'green' if profit_pct > 0 else 'red', 
                             'black', 'black', 'green' if fuzzy_profit_pct > 0 else 'red', 'black'], size=13),
            height=30
        )
    ), row=1, col=1
)

# --- Subplot 2, Links: RSI-Strategie (Zeile 2, Spalte 1) ---
fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Kurs', line=dict(color='#1A365D', width=2), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], mode='lines', name='SMA 20', line=dict(color='#F59E0B', dash='dot'), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=df.index, y=buy_signals, mode='markers', name='RSI Kauf',
                         marker=dict(color='green', size=12, symbol='triangle-up', line=dict(color='darkgreen', width=1))), row=2, col=1)
fig.add_trace(go.Scatter(x=df.index, y=sell_signals, mode='markers', name='RSI Verkauf',
                         marker=dict(color='red', size=12, symbol='triangle-down', line=dict(color='darkred', width=1))), row=2, col=1)

# --- Subplot 2, Rechts: Neue Fuzzy-Strategie (Zeile 2, Spalte 2) ---
fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Kurs', line=dict(color='#64748B', width=1.5), showlegend=False), row=2, col=2)
fig.add_trace(go.Scatter(x=df.index, y=fuzzy_buy_signals, mode='markers', name='Fuzzy Kauf',
                         marker=dict(color='#059669', size=12, symbol='hexagram', line=dict(color='black', width=1))), row=2, col=2)
fig.add_trace(go.Scatter(x=df.index, y=fuzzy_sell_signals, mode='markers', name='Fuzzy Verkauf',
                         marker=dict(color='#DC2626', size=12, symbol='x', line=dict(color='black', width=1))), row=2, col=2)

# --- Subplot 3: RSI (Zeile 3) ---
fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI', line=dict(color='#8B5CF6'), showlegend=False), row=3, col=1)
fig.add_trace(go.Scatter(x=df.index, y=[30]*len(df), mode='lines', name='30er Grenze', line=dict(color='green', dash='dash', width=1.5), showlegend=False, opacity=0.7), row=3, col=1)
fig.add_trace(go.Scatter(x=df.index, y=[50]*len(df), mode='lines', name='50er Grenze', line=dict(color='red', dash='dash', width=1.5), showlegend=False, opacity=0.7), row=3, col=1)

# --- Subplot 4: Signalstärke in % (Zeile 4) ---
fig.add_trace(
    go.Scatter(
        x=df.index, y=df['Signalstaerke_Pct'], mode='lines', name='Kaufsignal (%)', 
        line=dict(color='#10B981', width=2), fill='tozeroy', fillcolor='rgba(16, 185, 129, 0.15)', showlegend=False
    ), row=4, col=1
)

# --- Subplot 5: Fuzzy Gesamtscore (Zeile 5) ---
fig.add_trace(
    go.Scatter(
        x=df.index, y=df['Fuzzy_Gesamtscore'], mode='lines', name='Fuzzy Score (%)', 
        line=dict(color='#2563EB', width=2), fill='tozeroy', fillcolor='rgba(37, 99, 235, 0.15)', showlegend=False
    ), row=5, col=1
)

# Y-Achsen für Prozentwerte fixieren
fig.update_yaxes(range=[0, 105], row=4, col=1)
fig.update_yaxes(range=[0, 105], row=5, col=1)

# Layout anpassen (Helles Design)
fig.update_layout(
    template="plotly_white", hovermode='x unified', height=1250,
    title="RHM.DE Strategie-Vergleich ( nebeneinander )",
    plot_bgcolor='white', paper_bgcolor='white'
)

# fig.show()

fig.write_html("index.html")