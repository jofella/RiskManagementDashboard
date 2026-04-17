import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from util.data_utils import load_dax_index, get_log_returns

WINDOW = 252  # 1 trading year


def _rolling_var_normal(losses, alpha, window=WINDOW):
    var = np.full(len(losses), np.nan)
    for t in range(window, len(losses)):
        window_losses = losses[t - window:t]
        mu_w = np.mean(window_losses)
        sigma_w = np.std(window_losses, ddof=1)
        var[t] = mu_w + sigma_w * stats.norm.ppf(alpha)
    return var


def _rolling_var_hist(losses, alpha, window=WINDOW):
    var = np.full(len(losses), np.nan)
    for t in range(window, len(losses)):
        var[t] = np.percentile(losses[t - window:t], alpha * 100)
    return var


def render():
    st.markdown("""
    ### Value at Risk (VaR)

    VaR is the **α-quantile of the loss distribution**:

    $$\\text{VaR}_\\alpha(L) = F_L^{-1}(\\alpha) = \\inf\\{l \\in \\mathbb{R} : P(L > l) \\leq 1-\\alpha\\}$$

    **Interpretation:** With probability $\\alpha$, the loss over the next period will **not exceed**
    $\\text{VaR}_\\alpha$. Equivalently, there is a $(1-\\alpha)$ chance of exceeding it.
    """)

    with st.expander("Pros & Cons of VaR"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Advantages**
            - Intuitive single-number summary of downside risk
            - Regulatory standard (Basel II/III)
            - Easy to communicate to management
            """)
        with col2:
            st.markdown("""
            **Limitations**
            - Ignores the **severity** of losses beyond VaR (tail blindness)
            - Not subadditive in general → diversification can increase VaR
            - Sensitive to the estimation method and sample window
            """)

    st.write("---")

    data = load_dax_index()
    losses = -np.diff(data)  # daily price-level losses
    lr = get_log_returns(data)
    T = len(losses)
    t_axis = np.arange(T)

    col1, col2 = st.columns(2)
    with col1:
        alpha = st.select_slider(
            "Confidence level α:", options=[0.90, 0.95, 0.975, 0.99], value=0.99
        )
    with col2:
        method = st.radio("Estimation method:", ["Normal (parametric)", "Historical Simulation"], horizontal=True)

    st.write("---")

    # --- Parametric (full-sample) VaR for context ---
    mu_full = np.mean(losses)
    sigma_full = np.std(losses, ddof=1)
    var_static = mu_full + sigma_full * stats.norm.ppf(alpha)

    # --- Rolling VaR ---
    if method == "Normal (parametric)":
        var_rolling = _rolling_var_normal(losses, alpha)
        method_label = "Normal (parametric)"
    else:
        var_rolling = _rolling_var_hist(losses, alpha)
        method_label = "Historical Simulation"

    exceedances = np.where(
        (~np.isnan(var_rolling)) & (losses > var_rolling)
    )[0]

    # --- Metrics row ---
    n_valid = np.sum(~np.isnan(var_rolling))
    n_exceed = len(exceedances)
    exceed_pct = n_exceed / n_valid * 100 if n_valid > 0 else 0
    expected_pct = (1 - alpha) * 100

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Static VaR", f"{var_static:,.1f} pts")
    m2.metric("Exceedances", f"{n_exceed}")
    m3.metric("Exceedance rate", f"{exceed_pct:.2f}%")
    m4.metric("Expected rate", f"{expected_pct:.1f}%")

    st.write("---")

    # --- Rolling VaR explanation ---
    st.subheader(f"Rolling {WINDOW}-Day VaR ({method_label}, α={alpha})")
    st.caption(
        "Each day's VaR is estimated from the **previous 252 trading days** (one rolling year). "
        "Red dots mark days where the actual loss exceeded the predicted VaR (exceedances)."
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_axis, y=losses,
        mode="lines", name="Daily Loss", line=dict(color="steelblue", width=1), opacity=0.7
    ))
    fig.add_trace(go.Scatter(
        x=t_axis, y=var_rolling,
        mode="lines", name=f"VaR_{alpha}", line=dict(color="orange", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=exceedances, y=losses[exceedances],
        mode="markers", name="Exceedance",
        marker=dict(color="red", size=5, symbol="circle")
    ))
    fig.update_layout(
        xaxis_title="Trading Day", yaxis_title="Loss (index points)",
        legend=dict(x=0, y=1), xaxis=dict(showgrid=True), yaxis=dict(showgrid=True)
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Normal distribution illustration ---
    st.write("---")
    st.subheader("VaR on the Loss Distribution")
    st.caption("Illustration using full-sample parameters for the normal parametric method.")

    x_range = np.linspace(mu_full - 5 * sigma_full, mu_full + 5 * sigma_full, 500)
    pdf_vals = stats.norm.pdf(x_range, mu_full, sigma_full)

    fig2 = go.Figure()
    # Shade tail beyond VaR
    tail_mask = x_range >= var_static
    fig2.add_trace(go.Scatter(
        x=np.concatenate([[var_static], x_range[tail_mask], [x_range[tail_mask][-1]]]),
        y=np.concatenate([[0], pdf_vals[tail_mask], [0]]),
        fill="toself", fillcolor="rgba(220,50,50,0.3)",
        line=dict(color="rgba(0,0,0,0)"), name=f"Tail ({(1-alpha)*100:.1f}%)"
    ))
    fig2.add_trace(go.Scatter(
        x=x_range, y=pdf_vals,
        mode="lines", name="Loss PDF (Normal)", line=dict(color="steelblue", width=2)
    ))
    fig2.add_vline(
        x=var_static, line_dash="dash", line_color="crimson",
        annotation_text=f"VaR_{alpha} = {var_static:.1f}", annotation_position="top right"
    )
    fig2.update_layout(
        xaxis_title="Loss", yaxis_title="Density",
        legend=dict(x=0, y=1)
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown(f"""
    **Interpretation:** $\\text{{VaR}}_{{\\alpha={alpha}}} = {var_static:.1f}$ index points.
    On {(1-alpha)*100:.1f}% of trading days, the loss **exceeds** this threshold.
    The red shaded area represents the unmodelled **tail risk** — what happens beyond VaR
    is not captured. This is the key weakness that **Expected Shortfall (ES)** addresses.
    """)
