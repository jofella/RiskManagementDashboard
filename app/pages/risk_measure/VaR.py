import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from util.data_utils import load_dax_index, load_dax_companies, get_log_returns

WINDOW = 252  # 1 trading year


@st.cache_data
def _rolling_var_normal(losses, alpha, window=WINDOW):
    var = np.full(len(losses), np.nan)
    for t in range(window, len(losses)):
        window_losses = losses[t - window:t]
        mu_w = np.mean(window_losses)
        sigma_w = np.std(window_losses, ddof=1)
        var[t] = mu_w + sigma_w * stats.norm.ppf(alpha)
    return var


@st.cache_data
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

    with st.expander("📖 Quantile functions and generalised inverses (F←)"):
        st.markdown(r"""
        The VaR is formally defined as the **generalised inverse** (also called the quantile function)
        of the loss CDF $F_L$:

        $$\text{VaR}_\alpha(L) = F_L^\leftarrow(\alpha) = \inf\{l \in \mathbb{R} : F_L(l) \geq \alpha\}$$

        **Why the infimum?** For a continuous, strictly increasing CDF, $F^\leftarrow(\alpha)$ is just
        the unique $l$ with $F(l) = \alpha$. But if $F$ has **jumps** (e.g. discrete distributions)
        or **flat parts**, the inverse is not unique — we take the infimum to get a well-defined value.

        **Key properties:**
        - $F^\leftarrow(F(x)) \leq x$ always; equality holds **if** $F$ is continuous at $x$.
        - $F(F^\leftarrow(\alpha)) \geq \alpha$ always; equality holds **if** $F$ has no jump at $F^\leftarrow(\alpha)$.
        - For continuous $F$: $F^\leftarrow(F(x)) = x$ and $F(F^\leftarrow(\alpha)) = \alpha$ — both hold.

        **Practical implication:** When we say $\text{VaR}_{99\%} = q$, we mean $F_L(q) \geq 0.99$ and
        $F_L(q^-) < 0.99$. For continuous loss distributions (the common case in market risk),
        this simplifies to $F_L(\text{VaR}_\alpha) = \alpha$ exactly.

        **Connection to historical simulation:** The empirical quantile $\hat{F}_n^\leftarrow(\alpha)$
        is just the $\lceil n\alpha \rceil$-th order statistic $X_{(\lceil n\alpha \rceil)}$ —
        the smallest observation such that at least fraction $\alpha$ of the data lies below it.
        """)

    with st.expander("📖 Is VaR convex? Why does it matter?"):
        st.markdown(r"""
        A risk measure $\varrho$ is **convex** if for $\lambda \in [0,1]$:

        $$\varrho(\lambda L_1 + (1-\lambda) L_2) \leq \lambda \varrho(L_1) + (1-\lambda)\varrho(L_2)$$

        Convexity means *mixing two positions cannot increase risk above the weighted average* —
        a diversification principle.

        **VaR is generally NOT convex.** The same bond counterexample shows this:
        - $L_1$: lose €100 with probability 1%, zero otherwise. $\text{VaR}_{99\%}(L_1) = 0$.
        - $L_2 = L_1$ (same bond, identical risk). $\text{VaR}_{99\%}(L_2) = 0$.
        - Mix $\lambda = 0.5$: $0.5 L_1 + 0.5 L_2 = L_1$ (since $L_1 = L_2$). Still $\text{VaR}_{99\%} = 0$.

        But for two *different* independent bonds of the same type:
        - $\text{VaR}_{99\%}(0.5 L_1 + 0.5 L_2) > 0$ because $P(\text{at least one defaults}) \approx 2\%$.
        - Yet $0.5 \text{VaR}_{99\%}(L_1) + 0.5 \text{VaR}_{99\%}(L_2) = 0$.

        **Why it matters:** Non-convexity means a portfolio desk could split one position into two
        sub-positions and *reduce* reported VaR without actually reducing risk. This creates regulatory
        arbitrage and incentivises fragmentation of risk that VaR cannot detect.

        **ES is convex** (and subadditive), which is why it closes this loophole.
        """)

    with st.expander("📖 Intuition: VaR in plain language"):
        st.markdown(r"""
        **$\text{VaR}_{99\%} = €1\text{M}$** means: *"On 99 out of 100 trading days, we will
        not lose more than €1 million. On the remaining 1 day, we might lose more — and VaR
        tells us nothing about how much more."*

        Think of VaR as a **flood insurance threshold**, not a worst-case loss. If your house
        is in a "1-in-100-year flood zone", VaR tells you where the flood line is — but
        nothing about how deep the flood will be when it comes.

        **How is it used in practice?**
        - **Trading desks:** Daily P&L is monitored against VaR limits. Breach → position review.
        - **Basel II/III:** Banks must hold capital equal to a multiple of their 10-day 99% VaR.
        - **Internal risk reporting:** Senior management receives a daily one-number risk summary.

        **The estimation methods compared:**
        | Method | Data used | Assumption | Strength |
        |---|---|---|---|
        | Normal parametric | Mean + std of window | Returns are normal | Fast, analytical |
        | Historical simulation | Empirical quantile of window | No distribution assumption | Captures fat tails, skewness |
        | GARCH-based | Conditional variance model | Volatility clustering | Time-varying risk |
        """)

    with st.expander("📖 Deep dive: Why is VaR not subadditive?"):
        st.markdown(r"""
        A **coherent risk measure** (Artzner et al., 1999) must satisfy four axioms, one of which is
        **subadditivity**: $\varrho(L_A + L_B) \leq \varrho(L_A) + \varrho(L_B)$.

        Subadditivity captures the idea that *diversification cannot increase risk* — merging
        two portfolios should not require more capital than holding them separately.

        **VaR violates this.** A simple counterexample:
        - Bond A defaults with probability 1% → loses €100; otherwise zero.
        - Bond B: same, independent of A.
        - $\text{VaR}_{99\%}(A) = 0$ and $\text{VaR}_{99\%}(B) = 0$.
        - But $P(\text{at least one default}) \approx 2\%$, so $\text{VaR}_{99\%}(A+B) > 0$.

        So $\text{VaR}_{99\%}(A+B) > \text{VaR}_{99\%}(A) + \text{VaR}_{99\%}(B) = 0$.
        **Diversification increased the measured risk** — the opposite of economic intuition.

        This is why Basel IV (FRTB) replaced VaR with **Expected Shortfall** for internal
        model capital calculations. ES is always subadditive.
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

    # ------------------------------------------------------------------ #
    # Variance-Covariance Method (VCM) — multi-asset portfolio
    # ------------------------------------------------------------------ #
    st.write("---")
    st.subheader("3. Variance-Covariance Method (VCM)")
    st.markdown(r"""
    The **Variance-Covariance Method** applies VaR to a multi-asset portfolio by exploiting
    the **linearised loss**. For a portfolio with euro-denominated weights
    $\mathbf{w}_n = (\alpha_i S_{n,i})_i$, the linearised loss is:

    $$L^\Delta_{n+1} = -\mathbf{w}_n^\top X_{n+1}$$

    Assuming $X_{n+1} \mid \mathcal{F}_n \sim \mathcal{N}(\hat{\mu}_n,\, \hat{\Sigma}_n)$
    (multivariate normal with rolling estimates), the portfolio loss is also normal:

    $$L^\Delta_{n+1} \mid \mathcal{F}_n \;\sim\; \mathcal{N}\!\left(-\mathbf{w}_n^\top\hat{\mu}_n,\;
    \mathbf{w}_n^\top \hat{\Sigma}_n \mathbf{w}_n\right)$$

    so the VaR reduces to a closed-form expression:

    $$\text{VaR}_\alpha^{\text{VCM}} = -\mathbf{w}_n^\top \hat{\mu}_n +
    \sqrt{\mathbf{w}_n^\top \hat{\Sigma}_n \mathbf{w}_n}\cdot \Phi^{-1}(\alpha)$$

    The key advantage over single-asset VaR: the covariance matrix $\hat{\Sigma}_n$ captures
    **cross-asset correlations**, so diversification effects are automatically reflected.
    """)

    STOCK_NAMES_VCM = ["BMW", "SAP", "Volkswagen", "Continental", "Siemens"]
    data_comp = load_dax_companies()
    lr_comp = np.diff(np.log(data_comp), axis=0)
    T_comp = len(lr_comp)

    @st.cache_data
    def compute_vcm_var(lr_comp, data_comp, weights, confidence, window=WINDOW):
        n = len(lr_comp)
        var_out = np.full(n, np.nan)
        for t in range(window, n):
            w = weights * data_comp[t]
            window_lr = lr_comp[t - window:t]
            mu_w = np.mean(window_lr, axis=0)
            Sigma_w = np.cov(window_lr.T)
            port_mean = w @ mu_w
            port_var = w @ Sigma_w @ w
            var_out[t] = -port_mean + np.sqrt(max(port_var, 0)) * stats.norm.ppf(confidence)
        return var_out

    default_weights = np.array([4, 8, 15, 16, 23])
    var_vcm = compute_vcm_var(lr_comp, data_comp, default_weights, alpha)

    losses_port = np.array([
        -default_weights @ data_comp[t] * (np.exp(lr_comp[t]) - 1)
        for t in range(T_comp)
    ])

    vcm_valid = ~np.isnan(var_vcm)
    n_vcm_exceed = int(np.sum((losses_port > var_vcm) & vcm_valid))
    n_vcm_valid = int(np.sum(vcm_valid))

    m1, m2, m3 = st.columns(3)
    m1.metric("VCM exceedances", n_vcm_exceed)
    m2.metric("Exceedance rate", f"{n_vcm_exceed / n_vcm_valid * 100:.2f}%")
    m3.metric("Expected rate", f"{(1 - alpha) * 100:.1f}%")

    fig_vcm = go.Figure()
    fig_vcm.add_trace(go.Scatter(
        x=np.arange(T_comp), y=losses_port,
        mode="lines", name="Portfolio Loss (€)", line=dict(color="steelblue", width=1), opacity=0.6
    ))
    fig_vcm.add_trace(go.Scatter(
        x=np.arange(T_comp), y=var_vcm,
        mode="lines", name=f"VCM VaR_{alpha}", line=dict(color="orange", width=2)
    ))
    exc_vcm = np.where((losses_port > var_vcm) & vcm_valid)[0]
    fig_vcm.add_trace(go.Scatter(
        x=exc_vcm, y=losses_port[exc_vcm],
        mode="markers", name="Exceedance", marker=dict(color="red", size=5)
    ))
    fig_vcm.update_layout(
        xaxis_title="Trading Day", yaxis_title="Loss (€)",
        legend=dict(x=0, y=1), xaxis=dict(showgrid=True), yaxis=dict(showgrid=True)
    )
    st.plotly_chart(fig_vcm, use_container_width=True)

    st.markdown(f"""
    **Portfolio:** {dict(zip(STOCK_NAMES_VCM, default_weights.tolist()))} shares.
    The VCM uses the full **5×5 covariance matrix** of log returns, updated daily on a
    rolling 252-day window. Correlations between stocks reduce portfolio variance below
    the sum of individual variances — the mathematical expression of diversification.
    """)
