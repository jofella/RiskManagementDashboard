import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from util.data_utils import load_dax_companies

st.title("📉 Losses")
st.markdown("""
Financial institutions — banks, insurance companies, investment funds — are permanently exposed to
the risk of financial loss. A single extreme loss event can threaten not just the institution itself
but, through contagion, the broader financial system. This **systemic dimension** is why risk
management is subject to strict regulatory oversight (Basel II/III/IV for banks, Solvency II for
insurers) and why the quantification of losses is the foundation of all modern risk frameworks.

### Three Central Questions
Any rigorous risk management framework must answer:

1. **How to quantify risk?** — What is the appropriate mathematical object for measuring a firm's
   exposure? This section introduces the **loss operator**, which maps portfolio structure and
   market movements into a monetary loss figure.
2. **How to measure risk?** — Given a loss distribution, how do we summarise it in a single
   number suitable for capital calculations? This leads to **risk measures** (VaR, ES) in the
   next section.
3. **How much capital reserve is needed?** — Regulators require institutions to hold capital
   sufficient to absorb losses up to a specified confidence level. The risk measure directly
   determines this buffer.

### The General Framework
We model a portfolio as a function of **risk factors** $(Z_n)$ — typically log asset prices.
The **loss** over period $[t_n, t_{n+1}]$ is the negative change in portfolio value:

$$L_{n+1} = -(V_{n+1} - V_n) = \ell_{[n]}(X_{n+1})$$

where $X_{n+1} = Z_{n+1} - Z_n$ are the **risk factor changes** and $\ell_{[n]}$ is the
**loss operator** — a function known at time $t_n$. This separation between the
*portfolio structure* (encoded in $\ell_{[n]}$) and the *market movements* ($X_{n+1}$)
is the key modelling insight: it allows us to study the loss distribution independently
of specific portfolio choices.
""")
st.write("---")


# Load data
@st.cache_data
def load_data():
    return load_dax_companies()

data_dax_comp = load_data()

STOCK_NAMES = ["BMW", "SAP", "Volkswagen", "Continental", "Siemens"]

# 1. Loss Operator
st.header("1. Loss Operator")
st.markdown("""
### 1.1 Risk Factors and the Loss Operator

When analysing a stock portfolio, we choose **log stock prices** as **risk factors**:

$$Z_{n,i} := \\log(S_{n,i})$$

The **risk factor changes** (log returns) are:

$$X_{n+1} = Z_{n+1} - Z_n$$

Based on these, the **loss operator** computes portfolio losses as:

$$L_{n+1} = -(V_{n+1} - V_n) =: \\ell_{[n]}(X_{n+1})$$

The key advantage: we **separate the risk factors from the portfolio structure**,
making the modelling process more flexible.
""")
st.write("---")

# Portfolio weights
st.subheader("Portfolio Weights")
st.write("Set the number of shares held for each stock:")

cols = st.columns(5)
defaults = [4, 8, 15, 16, 23]
weights = []
for col, name, default in zip(cols, STOCK_NAMES, defaults):
    with col:
        weights.append(st.number_input(name, min_value=0, max_value=50, value=default, step=1))

alpha_weights = np.array(weights)


# Compute risk factors and losses
@st.cache_data
def compute_risk_factors(data):
    Z_n = np.log(data)
    X_n = np.diff(Z_n, axis=0)
    return Z_n, X_n


@st.cache_data
def compute_nonlinear_losses(X_n, alpha_weights, data):
    weighted_port = alpha_weights * data
    return np.array([-np.dot(weighted_port[n, :], np.exp(X_n[n, :]) - 1) for n in range(len(X_n))])


@st.cache_data
def compute_linearized_losses(X_n, alpha_weights, data):
    weighted_port = alpha_weights * data
    return np.array([-np.dot(weighted_port[n, :], X_n[n, :]) for n in range(len(X_n))])


Z_n, X_n = compute_risk_factors(data_dax_comp)
V_n = np.dot(np.exp(Z_n), alpha_weights)
losses = compute_nonlinear_losses(X_n, alpha_weights, data_dax_comp)
delta_losses = compute_linearized_losses(X_n, alpha_weights, data_dax_comp)

loss_df = pd.DataFrame({"Time": np.arange(len(losses)), "Losses": losses})
delta_loss_df = pd.DataFrame({"Time": np.arange(len(delta_losses)), "Losses": delta_losses})

st.write("---")

# Portfolio value chart
st.subheader("Portfolio Value Over Time")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=np.arange(len(V_n)), y=V_n,
    mode="lines", name="Portfolio Value", line=dict(color="steelblue")
))
fig.update_layout(
    title="DAX 5-Stock Portfolio (2000–Today)",
    xaxis_title="Time (Days)", yaxis_title="Portfolio Value (€)",
    xaxis=dict(showgrid=True), yaxis=dict(showgrid=True)
)
st.plotly_chart(fig, use_container_width=True)

# Portfolio losses chart
st.subheader("Portfolio Losses Over Time")
fig = px.line(
    loss_df, x="Time", y="Losses",
    labels={"Losses": "Loss (€)", "Time": "Time (Days)"},
    color_discrete_sequence=["steelblue"]
)
st.plotly_chart(fig, use_container_width=True)

st.write("---")

# 1.3 Linearized Loss
st.header("2. Linearized Loss Operator")
st.markdown("""
### 1.2 Linearization via Taylor Expansion

The loss operator is generally **nonlinear** due to the exponential function.
A **first-order Taylor expansion** gives the **linearized loss**:

$$\\tilde{L}_{n+1} = -\\sum_i \\alpha_i S_{n,i} \\cdot X_{n+1,i} = -\\mathbf{w}_n^\\top X_{n+1}$$

where $\\mathbf{w}_n = (\\alpha_i S_{n,i})_i$ are the **euro-denominated weights**.

This approximation is accurate for small risk factor changes and is the basis of the
**Variance-Covariance Method (VCM)** used in the next section.
""")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=loss_df["Time"], y=loss_df["Losses"],
    mode="lines", name="Nonlinear Loss", line=dict(color="steelblue")
))
fig.add_trace(go.Scatter(
    x=delta_loss_df["Time"], y=delta_loss_df["Losses"],
    mode="lines", name="Linearized Loss (δ)", line=dict(color="crimson", dash="dash")
))
fig.update_layout(
    title="Nonlinear vs. Linearized Portfolio Losses",
    xaxis_title="Time (Days)", yaxis_title="Loss (€)",
    legend=dict(x=0, y=1), xaxis=dict(showgrid=True), yaxis=dict(showgrid=True)
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
**Findings:** The linearized approximation closely tracks the true nonlinear loss for typical
daily moves. During extreme events (large $X_{n+1}$), the approximation slightly underestimates losses —
a known limitation of first-order linearization.
""")
