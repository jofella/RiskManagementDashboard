import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from util.data_utils import load_dax_index, get_log_returns

WINDOW = 252


@st.cache_data
def _rolling_var_normal(losses, alpha, window=WINDOW):
    var = np.full(len(losses), np.nan)
    for t in range(window, len(losses)):
        w = losses[t - window:t]
        var[t] = np.mean(w) + np.std(w, ddof=1) * stats.norm.ppf(alpha)
    return var


@st.cache_data
def _rolling_var_hist(losses, alpha, window=WINDOW):
    var = np.full(len(losses), np.nan)
    for t in range(window, len(losses)):
        var[t] = np.percentile(losses[t - window:t], alpha * 100)
    return var


@st.cache_data
def _rolling_es_normal(losses, alpha, window=WINDOW):
    es = np.full(len(losses), np.nan)
    for t in range(window, len(losses)):
        w = losses[t - window:t]
        mu_w = np.mean(w)
        sigma_w = np.std(w, ddof=1)
        # ES for normal: mu + sigma * phi(Phi^{-1}(alpha)) / (1 - alpha)
        es[t] = mu_w + sigma_w * stats.norm.pdf(stats.norm.ppf(alpha)) / (1 - alpha)
    return es


@st.cache_data
def _rolling_es_hist(losses, alpha, window=WINDOW):
    es = np.full(len(losses), np.nan)
    for t in range(window, len(losses)):
        w = losses[t - window:t]
        var_t = np.percentile(w, alpha * 100)
        tail = w[w > var_t]
        es[t] = np.mean(tail) if len(tail) > 0 else var_t
    return es


def render():
    st.markdown("""
    ### Expected Shortfall (ES)

    ES answers the question VaR deliberately ignores: **how bad is it when things go wrong?**

    $$\\text{ES}_\\alpha(L) = E[L \\mid L > \\text{VaR}_\\alpha(L)] = \\frac{1}{1-\\alpha} \\int_\\alpha^1 \\text{VaR}_u(L)\\, du$$

    For a **normally distributed** loss $L \\sim \\mathcal{N}(\\mu, \\sigma^2)$:

    $$\\text{ES}_\\alpha = \\mu + \\sigma \\cdot \\frac{\\phi(\\Phi^{-1}(\\alpha))}{1-\\alpha}$$

    where $\\phi$ is the standard normal PDF and $\\Phi^{-1}$ is the quantile function.
    """)

    with st.expander("Why ES is superior to VaR"):
        st.markdown("""
        | Property | VaR | ES |
        |---|---|---|
        | Captures tail severity | ✗ | ✓ |
        | Subadditive (diversification reduces risk) | ✗ in general | ✓ always |
        | Coherent risk measure | ✗ | ✓ |
        | Regulatory status | Basel II/III | Basel IV (FRTB) |
        | Estimation difficulty | Low | Moderate |

        **Subadditivity** means $\\text{ES}(L_1 + L_2) \\leq \\text{ES}(L_1) + \\text{ES}(L_2)$ —
        combining risks can only reduce, never increase, the total ES.
        VaR does not guarantee this, which is why ES replaced VaR in Basel IV.
        """)

    st.write("---")

    data = load_dax_index()
    losses = -np.diff(data)
    T = len(losses)
    t_axis = np.arange(T)

    col1, col2 = st.columns(2)
    with col1:
        alpha = st.select_slider(
            "Confidence level α:", options=[0.90, 0.95, 0.975, 0.99], value=0.975
        )
    with col2:
        method = st.radio("Estimation method:", ["Normal (parametric)", "Historical Simulation"], horizontal=True)

    st.write("---")

    mu_full = np.mean(losses)
    sigma_full = np.std(losses, ddof=1)
    var_static = mu_full + sigma_full * stats.norm.ppf(alpha)
    es_static = mu_full + sigma_full * stats.norm.pdf(stats.norm.ppf(alpha)) / (1 - alpha)

    if method == "Normal (parametric)":
        var_rolling = _rolling_var_normal(losses, alpha)
        es_rolling = _rolling_es_normal(losses, alpha)
        method_label = "Normal (parametric)"
    else:
        var_rolling = _rolling_var_hist(losses, alpha)
        es_rolling = _rolling_es_hist(losses, alpha)
        method_label = "Historical Simulation"

    exceedances_var = np.where((~np.isnan(var_rolling)) & (losses > var_rolling))[0]
    exceedances_es = np.where((~np.isnan(es_rolling)) & (losses > es_rolling))[0]

    n_valid = np.sum(~np.isnan(es_rolling))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Static VaR", f"{var_static:,.1f} pts")
    m2.metric("Static ES", f"{es_static:,.1f} pts")
    m3.metric("ES > VaR by", f"{es_static - var_static:,.1f} pts")
    m4.metric("ES exceedances", f"{len(exceedances_es)}")

    st.write("---")

    st.subheader(f"Rolling {WINDOW}-Day VaR vs ES ({method_label}, α={alpha})")
    st.caption(
        "ES (orange) always lies **above** VaR (green) — it accounts for the average loss in the tail, "
        "not just its threshold. Red dots are ES exceedances."
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_axis, y=losses,
        mode="lines", name="Daily Loss", line=dict(color="steelblue", width=1), opacity=0.6
    ))
    fig.add_trace(go.Scatter(
        x=t_axis, y=var_rolling,
        mode="lines", name=f"VaR_{alpha}", line=dict(color="limegreen", width=2, dash="dash")
    ))
    fig.add_trace(go.Scatter(
        x=t_axis, y=es_rolling,
        mode="lines", name=f"ES_{alpha}", line=dict(color="orange", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=exceedances_es, y=losses[exceedances_es],
        mode="markers", name="ES Exceedance",
        marker=dict(color="red", size=5, symbol="circle")
    ))
    fig.update_layout(
        xaxis_title="Trading Day", yaxis_title="Loss (index points)",
        legend=dict(x=0, y=1), xaxis=dict(showgrid=True), yaxis=dict(showgrid=True)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Distribution illustration
    st.write("---")
    st.subheader("VaR vs ES on the Loss Distribution")
    st.caption("Full-sample normal fit. ES is the conditional mean of all losses beyond VaR.")

    x_range = np.linspace(mu_full - 5 * sigma_full, mu_full + 6 * sigma_full, 600)
    pdf_vals = stats.norm.pdf(x_range, mu_full, sigma_full)

    fig2 = go.Figure()
    tail_mask = x_range >= var_static
    fig2.add_trace(go.Scatter(
        x=np.concatenate([[var_static], x_range[tail_mask], [x_range[tail_mask][-1]]]),
        y=np.concatenate([[0], pdf_vals[tail_mask], [0]]),
        fill="toself", fillcolor="rgba(220,50,50,0.25)",
        line=dict(color="rgba(0,0,0,0)"), name=f"Tail beyond VaR"
    ))
    fig2.add_trace(go.Scatter(
        x=x_range, y=pdf_vals,
        mode="lines", name="Loss PDF (Normal)", line=dict(color="steelblue", width=2)
    ))
    fig2.add_vline(
        x=var_static, line_dash="dash", line_color="limegreen",
        annotation_text=f"VaR = {var_static:.1f}", annotation_position="top left"
    )
    fig2.add_vline(
        x=es_static, line_dash="dash", line_color="orange",
        annotation_text=f"ES = {es_static:.1f}", annotation_position="top right"
    )
    fig2.update_layout(
        xaxis_title="Loss", yaxis_title="Density",
        legend=dict(x=0, y=1)
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown(f"""
    **Interpretation:** At confidence level $\\alpha = {alpha}$:
    - $\\text{{VaR}} = {var_static:.1f}$ pts — the threshold exceeded on {(1-alpha)*100:.1f}% of days
    - $\\text{{ES}} = {es_static:.1f}$ pts — the **average loss** on those bad days

    ES is always $\\geq$ VaR. The gap ({es_static - var_static:.1f} pts here) represents the
    **additional tail severity** that VaR ignores. In fat-tailed distributions this gap is much larger.
    """)
