import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from util.data_utils import load_dax_index, get_log_returns


def render():
    st.markdown("""
    ### Standard Deviation as a Risk Measure

    A classical approach to measuring risk uses the **standard deviation of the loss distribution**:

    $$\\varrho(L) = E_n(L_{n+1}) + c \\cdot \\sqrt{\\operatorname{Var}_n(L_{n+1})}$$

    where $c > 0$ is a **risk-appetite parameter** (e.g. $c = 1.64$ corresponds to the 95% normal quantile).

    For a single stock position with log-normally distributed returns $X \\sim \\mathcal{N}(\\mu, \\sigma^2)$,
    the conditional moments of the loss $L = -S_n(e^X - 1)$ are:

    $$E_n(L_{n+1}) = S_n \\left(1 - e^{\\mu + \\sigma^2/2}\\right)$$
    $$\\operatorname{Var}_n(L_{n+1}) = S_n^2 \\left(e^{\\sigma^2} - 1\\right) e^{2\\mu + \\sigma^2}$$
    """)

    with st.expander("Pros & Cons of Standard Deviation"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Advantages**
            - Simple and well-understood
            - Easy to estimate from data
            - Useful as a general volatility indicator
            """)
        with col2:
            st.markdown("""
            **Limitations**
            - Symmetric: treats gains and losses equally
            - Misses tail risk — does not focus on extreme losses
            - Undefined for heavy-tailed distributions with infinite variance
            """)

    st.write("---")

    data = load_dax_index()
    lr = get_log_returns(data)
    mu = np.mean(lr)
    sigma = np.std(lr)
    losses = -np.diff(data)

    c = st.slider("Risk-appetite parameter c:", min_value=0.5, max_value=3.0, value=1.64, step=0.01)
    st.caption(f"c = {c:.2f} corresponds roughly to the {100*float(st.session_state.get('_sd_alpha', 0)):.0f}% normal quantile" if False else "")

    def rho(S_n, c, mu, sigma):
        cond_mean = S_n * (1 - np.exp(mu + sigma**2 / 2))
        cond_var = S_n**2 * (np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2)
        return cond_mean + c * np.sqrt(cond_var)

    sd_risk = np.array([rho(data[i], c, mu, sigma) for i in range(len(data))])

    loss_df = pd.DataFrame({"Index": range(len(losses)), "Losses": losses})
    sd_df = pd.DataFrame({"Index": range(len(data)), "Risk Measure": sd_risk})

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=loss_df["Index"], y=loss_df["Losses"],
        mode="lines", name="Daily Losses", line=dict(color="steelblue", width=1)
    ))
    fig.add_trace(go.Scatter(
        x=sd_df["Index"], y=sd_df["Risk Measure"],
        mode="lines", name=f"ϱ (c={c:.2f})", line=dict(color="crimson", width=2)
    ))
    fig.update_layout(
        xaxis_title="Trading Day", yaxis_title="Loss (index points)",
        legend=dict(x=0, y=1), xaxis=dict(showgrid=True), yaxis=dict(showgrid=True)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Observation:** The risk measure grows with the DAX index level (since absolute losses scale
    with price), but it cannot adapt to **volatility regimes**. During crises, actual losses spike
    far above the estimated risk — motivating VaR and ES.
    """)
