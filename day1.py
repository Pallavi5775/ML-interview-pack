import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

# ----------------------
# 1. Generate FinX data
# ----------------------
np.random.seed(42)

# Normal: daily returns (in %)
returns = np.random.normal(loc=0.1, scale=1.5, size=1000)

# Binomial: profitable trades (out of 10)
trades = np.random.binomial(n=10, p=0.55, size=1000)

# Poisson: event counts (defaults per month)
defaults = np.random.poisson(lam=3, size=1000)

# Smooth density for Normal
x_norm = np.linspace(min(returns), max(returns), 200)
kde = gaussian_kde(returns)
y_norm = kde(x_norm)

# Frequency for Binomial
unique_bin, counts_bin = np.unique(trades, return_counts=True)
y_bin = counts_bin / len(trades)

# Frequency for Poisson
unique_poi, counts_poi = np.unique(defaults, return_counts=True)
y_poi = counts_poi / len(defaults)

# ----------------------
# 2. Create subplot layout
# ----------------------
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=(
        'Market Returns — Normal Distribution',
        'Trade Outcomes — Binomial Distribution',
        'Event Frequency — Poisson Distribution'
    )
)

# Normal
fig.add_trace(
    go.Scatter(
        x=x_norm, y=y_norm,
        mode='lines',
        name='Returns',
        line=dict(color='royalblue', dash='dot', width=3),
        hovertemplate='Return=%{x:.2f}%<br>Density=%{y:.3f}'
    ),
    row=1, col=1
)

# Binomial
fig.add_trace(
    go.Scatter(
        x=unique_bin, y=y_bin,
        mode='lines+markers',
        name='Profitable Trades',
        line=dict(color='limegreen', dash='dot', width=3),
        marker=dict(size=8, symbol='circle'),
        hovertemplate='Profitable Trades=%{x}<br>Probability=%{y:.3f}'
    ),
    row=1, col=2
)

# Poisson
fig.add_trace(
    go.Scatter(
        x=unique_poi, y=y_poi,
        mode='lines+markers',
        name='Defaults per Month',
        line=dict(color='darkorange', dash='dot', width=3),
        marker=dict(size=8, symbol='square'),
        hovertemplate='Defaults=%{x}<br>Probability=%{y:.3f}'
    ),
    row=1, col=3
)

# ----------------------
# 3. Layout & annotations
# ----------------------
fig.update_layout(
    title_text='FinX: How Statistics Power AI in Finance',
    title_x=0.5,
    template='plotly_white',
    showlegend=False,
    height=500,
    annotations=[
        dict(text='Market returns fluctuate smoothly → modeled by Normal Distribution',
             x=0.13, y=-0.25, showarrow=False, xref='paper', yref='paper', font=dict(size=12)),
        dict(text='Each trade = success/failure → Binomial world (strategy accuracy)',
             x=0.5, y=-0.25, showarrow=False, xref='paper', yref='paper', font=dict(size=12)),
        dict(text='Count of rare events (defaults, fraud) → Poisson behavior',
             x=0.86, y=-0.25, showarrow=False, xref='paper', yref='paper', font=dict(size=12))
    ]
)

fig.update_xaxes(title_text='Daily Return (%)', row=1, col=1)
fig.update_yaxes(title_text='Density', row=1, col=1)
fig.update_xaxes(title_text='Profitable Trades (out of 10)', row=1, col=2)
fig.update_yaxes(title_text='Probability', row=1, col=2)
fig.update_xaxes(title_text='Defaults per Month', row=1, col=3)
fig.update_yaxes(title_text='Probability', row=1, col=3)

# Show the figure
fig.show()
# For discrete data → count unique values → get probabilities.

# For continuous data → estimate smooth density (no unique counting).

# Your choice of visualization and test (t-test, chi-square, etc.) 
# depends directly on whether the data is continuous or discrete.