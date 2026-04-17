import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy import stats
from scipy.optimize import minimize

from util.data_utils import load_dax_index, get_log_returns

st.title("📈 Extreme Value Theory (EVT)")
st.markdown(r"""
Standard risk models based on the normal distribution systematically **underestimate the probability
and severity of extreme losses** — precisely the events that cause institutional failures.
Extreme Value Theory (EVT) addresses this by providing a mathematically rigorous framework for
modelling the *tail* of a distribution, without imposing assumptions on its centre.

### The Theoretical Foundation
The key result underpinning EVT is the **Pickands–Balkema–de Haan theorem** (a conditional analogue
of the Fisher–Tippett–Gnedenko theorem for maxima): for a wide class of distributions, the
conditional excess distribution above a sufficiently high threshold $u$ converges to a
**Generalized Pareto Distribution (GPD)**:

$$P(L - u \leq y \mid L > u) \xrightarrow{u \to \infty} G_{\xi, \sigma}(y) = 1 - \left(1 + \frac{\xi y}{\sigma}\right)^{-1/\xi}$$

The two parameters carry the key information:
- **Shape $\xi$ (tail index):** Controls tail heaviness. $\xi > 0$ implies a **Pareto-type heavy tail**
  with polynomial decay; $\xi = 0$ gives an exponential tail (normal, lognormal);
  $\xi < 0$ gives a bounded tail. For financial returns, $\xi \in (0.2, 0.4)$ is typical.
- **Scale $\sigma > 0$:** Controls the spread of exceedances above the threshold.

### Two Main EVT Approaches
1. **Block Maxima (BM):** Fit a Generalised Extreme Value (GEV) distribution to the maxima of
   non-overlapping blocks (e.g. annual maxima). Intuitive but wasteful — discards all
   non-maxima observations.
2. **Peak-over-Threshold (POT):** Fit a GPD to all observations exceeding a high threshold $u$.
   More data-efficient and the preferred approach in practice.

### Why EVT Matters for Risk Management
Once the GPD is fitted, VaR and ES can be **extrapolated beyond the historical sample** to
confidence levels (e.g. 99.9%) where no historical data exists. This is critical for:
- **Regulatory stress testing** (Basel IV FRTB uses 97.5% ES)
- **Solvency II** (insurance requires the 99.5% VaR over a one-year horizon)
- **Economic capital** models for operational and credit risk

The EVT estimates are systematically **larger** than normal-distribution estimates at high
confidence levels — precisely because they account for the fat tail the normal ignores.
""")
st.write("---")


# --- Data ---
data = load_dax_index()
lr = get_log_returns(data)
losses = -lr  # log-return losses (positive = loss)
losses_pos = losses.copy()  # keep sign; negative values = gains


# ============================================================
# SECTION 1: QQ Plots
# ============================================================
st.header("1. Heavy Tails: QQ Plots")
st.markdown("""
A **Quantile-Quantile (QQ) plot** compares the empirical quantiles of the data to the theoretical
quantiles of a reference distribution. If the data follows the reference distribution, the points
lie on a straight line.

**Deviations in the upper-right tail** indicate **heavier tails** than the reference.
""")

dist_choice = st.radio(
    "Reference distribution:", ["Normal", "Student-t (ν=4)", "Student-t (ν=6)"], horizontal=True
)

sorted_losses = np.sort(losses)
n = len(sorted_losses)
probs = (np.arange(1, n + 1) - 0.5) / n

if dist_choice == "Normal":
    mu_fit, sigma_fit = np.mean(losses), np.std(losses, ddof=1)
    theoretical_q = stats.norm.ppf(probs, mu_fit, sigma_fit)
    ref_label = "Normal"
elif dist_choice == "Student-t (ν=4)":
    df = 4
    scale = np.std(losses, ddof=1) * np.sqrt((df - 2) / df)
    theoretical_q = stats.t.ppf(probs, df, loc=np.mean(losses), scale=scale)
    ref_label = "t(4)"
else:
    df = 6
    scale = np.std(losses, ddof=1) * np.sqrt((df - 2) / df)
    theoretical_q = stats.t.ppf(probs, df, loc=np.mean(losses), scale=scale)
    ref_label = "t(6)"

fig_qq = go.Figure()
fig_qq.add_trace(go.Scatter(
    x=theoretical_q, y=sorted_losses,
    mode="markers", name="Empirical quantiles",
    marker=dict(color="steelblue", size=3, opacity=0.5)
))
lim = max(abs(theoretical_q.min()), abs(theoretical_q.max()), abs(sorted_losses.min()), abs(sorted_losses.max()))
fig_qq.add_trace(go.Scatter(
    x=[-lim, lim], y=[-lim, lim],
    mode="lines", name="45° reference line", line=dict(color="crimson", dash="dash", width=2)
))
fig_qq.update_layout(
    xaxis_title=f"Theoretical Quantiles ({ref_label})",
    yaxis_title="Empirical Quantiles (DAX log-return losses)",
    xaxis=dict(showgrid=True), yaxis=dict(showgrid=True)
)
st.plotly_chart(fig_qq, use_container_width=True)

st.markdown("""
**Reading the plot:** Points **above** the 45° line in the upper-right mean the empirical tail
is **heavier** than the reference. The normal distribution shows the strongest deviation;
the t-distribution with low degrees of freedom fits better — confirming fat tails.
""")
st.write("---")


# ============================================================
# SECTION 2: Hill Estimator
# ============================================================
st.header("2. Hill Estimator: Tail Index")
st.markdown(r"""
The **Hill estimator** estimates the **tail index** $\xi$ of a Pareto-type tail:

$$\hat{\xi}_k = \frac{1}{k} \sum_{i=1}^{k} \log\frac{X_{(n-i+1)}}{X_{(n-k)}}$$

where $X_{(1)} \leq \cdots \leq X_{(n)}$ are the order statistics and $k$ is the number of
upper-order statistics used (the **tail threshold**).

- $\xi > 0$: **heavy tail** (Pareto-type) — the distribution has finite moments only up to $1/\xi$
- $\xi = 0$: **light tail** (exponential decay, e.g. normal/lognormal)
- The **Hill plot** shows $\hat{\xi}_k$ vs $k$; a stable plateau suggests a reliable estimate
""")

@st.cache_data
def compute_hill(losses):
    pos = np.sort(losses[losses > 0])[::-1]
    k_min, k_max = 10, min(500, len(pos) - 1)
    k_vals = np.arange(k_min, k_max + 1)
    estimates = np.array([np.mean(np.log(pos[:k] / pos[k])) for k in k_vals])
    return k_vals, estimates, k_min, k_max


# Only use positive losses (right tail = actual losses)
pos_losses = np.sort(losses[losses > 0])[::-1]  # descending order
n_pos = len(pos_losses)

k_values, hill_estimates, k_min, k_max = compute_hill(losses)

fig_hill = go.Figure()
fig_hill.add_trace(go.Scatter(
    x=k_values, y=hill_estimates,
    mode="lines", name="Hill estimator ξ̂", line=dict(color="steelblue", width=2)
))
fig_hill.add_hline(y=0, line_dash="dot", line_color="grey")
fig_hill.update_layout(
    xaxis_title="k (number of upper order statistics)",
    yaxis_title="ξ̂ (tail index estimate)",
    xaxis=dict(showgrid=True), yaxis=dict(showgrid=True)
)
st.plotly_chart(fig_hill, use_container_width=True)

# Read off estimate around k=100
k_ref = st.slider("Read off ξ at k =", min_value=k_min, max_value=k_max, value=100, step=5)
xi_at_k = hill_estimates[k_ref - k_min]
st.metric(f"ξ̂(k={k_ref})", f"{xi_at_k:.4f}")

st.markdown(f"""
**Interpretation:** $\\hat{{\\xi}} \\approx {xi_at_k:.3f}$ indicates a **heavy-tailed distribution**.
Finite moments exist only up to order $1/\\hat{{\\xi}} \\approx {1/xi_at_k:.1f}$.
In particular, the distribution has finite variance ($\\xi < 0.5$) but potentially heavy kurtosis.
""")
st.write("---")


# ============================================================
# SECTION 3: Mean Excess Plot (MEP)
# ============================================================
st.header("3. Mean Excess Plot (MEP)")
st.markdown(r"""
The **mean excess function** is:

$$e(u) = E[L - u \mid L > u]$$

For a **Generalized Pareto Distribution (GPD)** with shape $\xi > 0$, $e(u)$ is **linear** in $u$:

$$e(u) = \frac{\sigma + \xi u}{1 - \xi}$$

The **MEP** plots the empirical mean excess $\hat{e}(u)$ against the threshold $u$.
A **linear, upward-sloping** section indicates the start of the Pareto tail — use that region
as the threshold for the POT method.
""")

# Compute MEP
@st.cache_data
def compute_mep(losses):
    u_grid = np.percentile(losses[losses > 0], np.linspace(50, 98, 80))
    me = []
    for u in u_grid:
        exc = losses[losses > u] - u
        me.append(np.mean(exc) if len(exc) >= 5 else np.nan)
    me = np.array(me)
    return u_grid, me, ~np.isnan(me)


u_grid, me_vals, valid_me = compute_mep(losses)

fig_mep = go.Figure()
fig_mep.add_trace(go.Scatter(
    x=u_grid[valid_me], y=me_vals[valid_me],
    mode="lines+markers", name="Empirical e(u)",
    line=dict(color="steelblue"), marker=dict(size=4)
))
fig_mep.update_layout(
    xaxis_title="Threshold u (log-return loss)",
    yaxis_title="Mean excess e(u)",
    xaxis=dict(showgrid=True), yaxis=dict(showgrid=True)
)
st.plotly_chart(fig_mep, use_container_width=True)

st.markdown("""
**Reading the plot:** Look for the region where $e(u)$ becomes **approximately linear**.
That point is the optimal threshold $u^*$ for the POT method.
A strongly upward slope confirms a heavy (Pareto) tail.
""")
st.write("---")


# ============================================================
# SECTION 4: POT Method — GPD Fitting
# ============================================================
st.header("4. Peak-over-Threshold (POT) Method")
st.markdown(r"""
For losses exceeding a high threshold $u$, the **Pickands–Balkema–de Haan theorem** states that
the excess distribution converges to a **Generalized Pareto Distribution (GPD)**:

$$F_u(y) = P(L - u \leq y \mid L > u) \approx G_{\xi, \sigma}(y) = 1 - \left(1 + \frac{\xi y}{\sigma}\right)^{-1/\xi}$$

where $\xi$ is the **shape** (tail index) and $\sigma > 0$ is the **scale** parameter.

Once fitted, the tail CDF, VaR, and ES can be extrapolated to any confidence level.
""")

# Threshold selector
q_threshold = st.slider(
    "Threshold percentile u (applied to all losses):",
    min_value=85, max_value=99, value=95, step=1,
    help="Losses above this percentile are used to fit the GPD."
)
u_threshold = np.percentile(losses, q_threshold)
exceedances_pot = losses[losses > u_threshold] - u_threshold
n_u = len(exceedances_pot)
n_total = len(losses)

st.write(f"Threshold $u$ = {u_threshold:.5f} ({q_threshold}th percentile) → **{n_u} exceedances**")

if n_u < 20:
    st.warning("Too few exceedances for reliable GPD fitting. Lower the threshold percentile.")
else:
    # MLE fit of GPD using scipy
    # GPD in scipy: genpareto, with loc=0 fixed
    try:
        xi_fit, loc_fit, sigma_fit = stats.genpareto.fit(exceedances_pot, floc=0)

        col1, col2, col3 = st.columns(3)
        col1.metric("Shape ξ (tail index)", f"{xi_fit:.4f}")
        col2.metric("Scale σ", f"{sigma_fit:.5f}")
        col3.metric("Exceedances used", f"{n_u}")

        st.write("---")

        # GPD fit vs empirical on exceedances
        st.subheader("GPD Fit on Exceedances")
        y_grid = np.linspace(0, np.percentile(exceedances_pot, 99), 200)
        gpd_pdf = stats.genpareto.pdf(y_grid, xi_fit, loc=0, scale=sigma_fit)

        fig_gpd = go.Figure()
        fig_gpd.add_trace(go.Histogram(
            x=exceedances_pot, nbinsx=40,
            histnorm="probability density",
            name="Empirical exceedances", marker_color="steelblue", opacity=0.7
        ))
        fig_gpd.add_trace(go.Scatter(
            x=y_grid, y=gpd_pdf,
            mode="lines", name="Fitted GPD", line=dict(color="crimson", width=2)
        ))
        fig_gpd.update_layout(
            xaxis_title="Excess over threshold u", yaxis_title="Density"
        )
        st.plotly_chart(fig_gpd, use_container_width=True)

        # Tail VaR and ES from GPD
        st.write("---")
        st.subheader("EVT-Based VaR and ES")
        st.markdown(r"""
        From the fitted GPD, for any confidence level $\alpha > q_u$ (the threshold percentile):

        $$\text{VaR}_\alpha^{\text{EVT}} = u + \frac{\sigma}{\xi}\left[\left(\frac{n(1-\alpha)}{n_u}\right)^{-\xi} - 1\right]$$

        $$\text{ES}_\alpha^{\text{EVT}} = \frac{\text{VaR}_\alpha}{1-\xi} + \frac{\sigma - \xi u}{1-\xi}$$
        """)

        alphas_evt = [0.95, 0.975, 0.99, 0.999]
        evt_rows = []
        for a in alphas_evt:
            if a <= q_threshold / 100:
                evt_rows.append({"α": a, "VaR (EVT)": "—", "ES (EVT)": "—",
                                 "VaR (Normal)": "—", "ES (Normal)": "—"})
                continue
            # EVT VaR
            if xi_fit != 0:
                var_evt = u_threshold + (sigma_fit / xi_fit) * (
                    ((n_total * (1 - a)) / n_u) ** (-xi_fit) - 1
                )
            else:
                var_evt = u_threshold - sigma_fit * np.log((n_total * (1 - a)) / n_u)
            # EVT ES
            es_evt = (var_evt + sigma_fit - xi_fit * u_threshold) / (1 - xi_fit)

            # Normal benchmark
            mu_n, sigma_n = np.mean(losses), np.std(losses, ddof=1)
            var_norm = mu_n + sigma_n * stats.norm.ppf(a)
            es_norm = mu_n + sigma_n * stats.norm.pdf(stats.norm.ppf(a)) / (1 - a)

            evt_rows.append({
                "α": a,
                "VaR (EVT)": f"{var_evt:.5f}",
                "ES (EVT)": f"{es_evt:.5f}",
                "VaR (Normal)": f"{var_norm:.5f}",
                "ES (Normal)": f"{es_norm:.5f}",
            })

        st.dataframe(pd.DataFrame(evt_rows).set_index("α"), use_container_width=True)

        st.markdown(f"""
        **Key finding:** EVT-based estimates are **larger** than normal estimates at high confidence
        levels — reflecting the fat tail that the normal distribution underestimates. The difference
        grows with $\\alpha$ and is most pronounced at extreme levels like $\\alpha = 0.999$
        (the "1-in-1000" day).
        """)

        # Tail CDF plot
        st.write("---")
        st.subheader("Tail CDF: Empirical vs GPD vs Normal")

        # Empirical survival function for upper tail
        sorted_all = np.sort(losses)[::-1]
        empirical_sf = np.arange(1, len(sorted_all) + 1) / len(sorted_all)
        tail_mask = sorted_all > u_threshold

        x_tail = np.linspace(u_threshold, sorted_all[0] * 1.2, 200)
        gpd_sf = (n_u / n_total) * stats.genpareto.sf(x_tail - u_threshold, xi_fit, loc=0, scale=sigma_fit)
        norm_sf = stats.norm.sf(x_tail, np.mean(losses), np.std(losses, ddof=1))

        fig_tail = go.Figure()
        fig_tail.add_trace(go.Scatter(
            x=sorted_all[tail_mask], y=empirical_sf[tail_mask],
            mode="markers", name="Empirical tail",
            marker=dict(color="steelblue", size=4, opacity=0.6)
        ))
        fig_tail.add_trace(go.Scatter(
            x=x_tail, y=gpd_sf,
            mode="lines", name="GPD (EVT)", line=dict(color="crimson", width=2)
        ))
        fig_tail.add_trace(go.Scatter(
            x=x_tail, y=norm_sf,
            mode="lines", name="Normal", line=dict(color="orange", width=2, dash="dash")
        ))
        fig_tail.add_vline(
            x=u_threshold, line_dash="dot", line_color="grey",
            annotation_text=f"u = {u_threshold:.4f}", annotation_position="top right"
        )
        fig_tail.update_layout(
            xaxis_title="Loss (log-return)",
            yaxis_title="P(L > x) — Survival probability",
            yaxis_type="log",
            xaxis=dict(showgrid=True), yaxis=dict(showgrid=True)
        )
        st.plotly_chart(fig_tail, use_container_width=True)

        st.markdown("""
        **Log-scale tail plot:** The GPD (red) hugs the empirical tail much more closely than
        the Normal (orange), which underestimates extreme probabilities. This is why EVT-based
        VaR and ES estimates are larger — and more realistic — than normal-distribution estimates
        at high confidence levels.
        """)

    except Exception as e:
        st.error(f"GPD fitting failed: {e}. Try adjusting the threshold.")
