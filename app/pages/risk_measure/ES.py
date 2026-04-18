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

    with st.expander("📖 Three representations of ES — and which to use when"):
        st.markdown(r"""
        ES has three equivalent representations for **continuous** loss distributions:

        **1. Conditional expectation (most intuitive):**
        $$\text{ES}_\alpha(L) = E[L \mid L > \text{VaR}_\alpha(L)]$$
        *"The expected loss given that VaR is exceeded."* Easy to explain to management.

        **2. Integral of VaR (most useful analytically):**
        $$\text{ES}_\alpha(L) = \frac{1}{1-\alpha} \int_\alpha^1 \text{VaR}_u(L)\, du$$
        *"ES is the average of all VaR levels above $\alpha$."* This connects ES to the entire
        upper tail of the loss distribution, not just the conditional mean. Useful for proving
        subadditivity and for computing ES analytically from a known distribution.

        **3. Worst-case expectation (most theoretical):**
        $$\text{ES}_\alpha(L) = \sup_{Q \ll P,\, \frac{dQ}{dP} \leq \frac{1}{1-\alpha}} E^Q[L]$$
        *"ES is the maximum expected loss over all scenarios that weight no event more than
        $1/(1-\alpha)$ times its physical probability."* This is the **generalised scenarios**
        interpretation and connects ES to the theory of coherent risk measures (Artzner et al.).

        | Representation | Best used for... |
        |---|---|
        | Conditional expectation | Intuitive explanation, backtesting |
        | Integral of VaR | Analytical computation, proving subadditivity |
        | Worst-case expectation | Theoretical justification, robust risk measures |
        """)

    with st.expander("📖 ES for discontinuous distributions — why the definition must change"):
        st.markdown(r"""
        For a **continuous** loss distribution, $P(L = \text{VaR}_\alpha) = 0$, so the
        conditional expectation $E[L \mid L > \text{VaR}_\alpha]$ is well-defined and equals ES.

        For a **discrete or mixed distribution** (e.g. credit losses with point mass at zero),
        the CDF may have a jump at $\text{VaR}_\alpha$. This creates a problem:

        *Example:* Suppose $L = 0$ with probability 99% and $L = 100$ with probability 1%.
        At $\alpha = 98\%$, $\text{VaR}_{98\%} = 0$. But $P(L > 0) = 1\% < 2\% = 1-\alpha$,
        so the conditional event $\{L > 0\}$ captures only 1% of probability mass, not 2%.
        The naive formula $E[L \mid L > \text{VaR}_\alpha] = 100$ is correct here,
        but in other discrete cases it double-counts mass at the jump point.

        **The correct general definition** uses the integral representation:
        $$\text{ES}_\alpha(L) = \frac{1}{1-\alpha} \int_\alpha^1 F_L^\leftarrow(u)\, du$$
        This always gives the correct answer regardless of continuity, because it integrates
        over quantile levels $u \in [\alpha, 1]$ — sweeping through the tail uniformly.

        **Practical implication:** When computing ES empirically (historical simulation),
        you should average the losses *strictly above* VaR plus a fraction of the loss *at* VaR
        to exactly fill the remaining probability mass to $1-\alpha$. Most implementations
        ignore this subtlety at the cost of small errors near discrete mass points.
        """)

    with st.expander("📖 Intuition: ES in plain language"):
        st.markdown(r"""
        If VaR is the **flood line**, ES is the **average flood depth above that line**.

        Formally, ES at level $\alpha$ is the average loss *given* that the loss exceeds
        $\text{VaR}_\alpha$. It asks: "On the bad days, how bad are they on average?"

        **Concrete example:** Suppose daily losses (in €M) are: 0.1, 0.2, 0.3, …, 1.0.
        - $\text{VaR}_{90\%} = €0.9\text{M}$ — the 90th percentile.
        - $\text{ES}_{90\%} = €0.95\text{M}$ — average of the worst 10%: $(0.9 + 1.0)/2$.

        Now replace that €1.0M loss with a €100M catastrophe:
        - $\text{VaR}_{90\%}$ is **unchanged** — it only looks at the 90th percentile.
        - $\text{ES}_{90\%}$ jumps to **€50.45M** — it correctly reflects the severity.

        **This is the core message:** VaR is blind to what happens beyond its threshold.
        ES is not. For a risk manager, the question is never just "will we breach VaR?" but
        "how much will we lose if we do?" — and only ES answers this.
        """)

    with st.expander("📖 Coherence: the four axioms"):
        st.markdown(r"""
        A risk measure $\varrho$ is **coherent** (Artzner, Delbaen, Eber, Heath 1999) if it satisfies:

        1. **Translation invariance:** $\varrho(L + c) = \varrho(L) + c$ — adding a sure loss of $c$
           increases risk by exactly $c$.
        2. **Positive homogeneity:** $\varrho(\lambda L) = \lambda \varrho(L)$ for $\lambda > 0$ —
           doubling a position doubles its risk.
        3. **Monotonicity:** $L_1 \leq L_2 \Rightarrow \varrho(L_1) \leq \varrho(L_2)$ — larger
           losses imply higher risk.
        4. **Subadditivity:** $\varrho(L_1 + L_2) \leq \varrho(L_1) + \varrho(L_2)$ — diversification
           cannot increase risk.

        **VaR fails subadditivity** (as shown in the VaR section). **ES satisfies all four**,
        making it the preferred regulatory risk measure under Basel IV. The practical consequence:
        under ES, a bank that merges two trading desks can never be required to hold more capital
        than the sum of the two desks' individual capital requirements.
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

    st.markdown(fr"""
    **Static estimates at $\alpha = {alpha}$:**
    - $\text{{VaR}}_{{{alpha}}} = {var_static:.1f}$ pts — the threshold breached on {(1-alpha)*100:.1f}% of days
    - $\text{{ES}}_{{{alpha}}} = {es_static:.1f}$ pts — the average loss on those bad days
    - The gap of **{es_static - var_static:.1f} pts** represents the tail severity VaR ignores

    **VaR vs ES ratio:** Under normality, $\text{{ES}}_\alpha / \text{{VaR}}_\alpha \approx
    \phi(\Phi^{{-1}}(\alpha)) / ((1-\alpha) \Phi^{{-1}}(\alpha))$, which approaches 1 as
    $\alpha \to 1$ for the normal but diverges for fat-tailed distributions. For the DAX
    returns, the empirical ES/VaR ratio is substantially higher than the normal prediction
    — reflecting the heavy tail that normal-based methods miss. This divergence is larger at
    higher confidence levels and is directly observable in the rolling chart above: during
    crisis periods, ES (orange) rises much more steeply than VaR (green), because the
    *average* severity of bad days increases with volatility clustering.

    **Dynamic risk management:** A risk manager observing the rolling chart should note that
    both VaR and ES spike sharply during 2008–2009 and 2020 — meaning the capital requirement
    implied by ES rises far above the static estimate during stress. This is why **dynamic
    estimation** (GARCH-conditional VaR and ES) is more conservative and more accurate than
    rolling-window methods: it captures volatility regime changes faster, updating the risk
    estimate daily rather than waiting for crisis observations to enter the rolling window.
    """)
