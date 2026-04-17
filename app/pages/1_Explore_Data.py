import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from util.data_utils import load_dax_index, get_log_returns

# --- HEADER ---
st.title("📊 Explore Data")
st.markdown("""
This section analyses **log returns** of the DAX index, compares them with a **normal distribution**,
and simulates a price process using Monte Carlo methods.
""")
st.write("---")

# 1. Load Data
st.header("1. Load Data")
st.write("We use DAX daily closing prices (2000–2024) to compute log returns.")

data = load_dax_index()
lr = get_log_returns(data)

mu = np.mean(lr)
sigma = np.std(lr)

st.success(f"Loaded **{len(data):,}** daily price observations → **{len(lr):,}** log returns.")
st.write("---")

# 2. Summary Statistics
st.header("2. Summary Statistics")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Mean (μ)", f"{mu:.6f}")
col2.metric("Std Dev (σ)", f"{sigma:.6f}")
col3.metric("Skewness", f"{stats.skew(lr):.4f}")
col4.metric("Excess Kurtosis", f"{stats.kurtosis(lr):.4f}")

st.markdown("""
> **Excess kurtosis > 0** is the fingerprint of **fat tails** — extreme returns occur more often
> than a normal distribution predicts. This is the core motivation for advanced risk models.
""")
st.write("---")

# 3. Log Returns Over Time
st.header("3. Log Returns Over Time")

sim_returns = np.random.normal(mu, sigma, len(lr))
S_t = data[0] * np.exp(np.cumsum(sim_returns))

lr_df = pd.DataFrame({"Index": range(len(lr)), "Log Returns": lr})
sim_df = pd.DataFrame({"Index": range(len(sim_returns)), "Simulated Log Returns": sim_returns})
price_df = pd.DataFrame({"Index": range(len(data)), "Real DAX": data})
sim_price_df = pd.DataFrame({"Index": range(len(S_t)), "Simulated DAX": S_t})

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=lr_df["Index"], y=lr_df["Log Returns"],
    mode="lines", name="Empirical", line=dict(color="steelblue")
))
fig.add_trace(go.Scatter(
    x=sim_df["Index"], y=sim_df["Simulated Log Returns"],
    mode="lines", name="Simulated (Normal)", line=dict(color="crimson", dash="dash")
))
fig.update_layout(
    xaxis_title="Trading Day", yaxis_title="Log Return",
    legend=dict(x=0, y=1), xaxis=dict(showgrid=True), yaxis=dict(showgrid=True)
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
**Findings:** Empirical returns show **volatility clustering** (calm periods followed by turbulent ones)
and extreme spikes not present in the stationary normal simulation — motivation for GARCH-type models.
""")
st.write("---")

# 4. Distribution of Log Returns
st.header("4. Distribution of Log Returns")

x_range = np.linspace(min(lr), max(lr), 300)
norm_pdf = stats.norm.pdf(x_range, loc=mu, scale=sigma)

num_bins = st.slider("Number of bins:", 20, 500, 250, step=10)

fig = px.histogram(
    lr_df, x="Log Returns", nbins=num_bins,
    histnorm="probability density",
    color_discrete_sequence=["steelblue"],
    opacity=0.7
)
fig.add_trace(go.Scatter(
    x=x_range, y=norm_pdf,
    mode="lines", name="Normal PDF", line=dict(color="crimson", width=2)
))
fig.update_layout(xaxis_title="Log Return", yaxis_title="Density")
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
**Findings:** The empirical distribution has **heavier tails** than the fitted normal —
extreme losses occur far more frequently than the bell curve predicts.
This motivates **VaR, ES, and Extreme Value Theory** as proper risk tools.
""")
st.write("---")

# 5. Monte Carlo Price Simulation
st.header("5. Monte Carlo Price Simulation")

st.info("This plot changes on every re-load since the simulated path is random.", icon="ℹ️")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=price_df["Index"], y=price_df["Real DAX"],
    mode="lines", name="Real DAX", line=dict(color="steelblue")
))
fig.add_trace(go.Scatter(
    x=sim_price_df["Index"], y=sim_price_df["Simulated DAX"],
    mode="lines", name="Simulated (GBM)", line=dict(color="crimson", dash="dash")
))
fig.update_layout(
    xaxis_title="Trading Day", yaxis_title="DAX Level",
    xaxis=dict(showgrid=True), yaxis=dict(showgrid=True)
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
**Findings:**
- The **simulated path** (Geometric Brownian Motion) uses constant volatility — unrealistic.
- The **real DAX** shows heteroscedasticity: volatility surges during crises (2008, 2020).
- Improvement: **GARCH(1,1)** allows time-varying volatility to better capture market dynamics.
""")
