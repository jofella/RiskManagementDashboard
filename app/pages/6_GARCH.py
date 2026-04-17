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

st.title("📉 GARCH(1,1) — Volatility Modelling")
st.markdown(r"""
The most important stylised fact of financial returns is **volatility clustering**: large moves
tend to be followed by large moves, and calm periods cluster together. The i.i.d. normal model
completely ignores this — it assumes constant volatility at all times. The
**Generalised AutoRegressive Conditional Heteroskedasticity** model of Bollerslev (1986)
resolves this by making volatility a dynamic, time-varying process.

### The GARCH(1,1) Model
Log returns are modelled as:

$$X_t = \sigma_t \varepsilon_t, \qquad \varepsilon_t \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0,1)$$

where the **conditional variance** $\sigma_t^2$ follows the recursion:

$$\sigma_t^2 = \omega + \alpha X_{t-1}^2 + \beta \sigma_{t-1}^2$$

- $\omega > 0$: long-run variance floor
- $\alpha \geq 0$: **ARCH effect** — how much yesterday's shock increases today's variance
- $\beta \geq 0$: **GARCH effect** — persistence of variance from one period to the next
- **Stationarity condition:** $\alpha + \beta < 1$; the unconditional variance is $\sigma^2_\infty = \omega / (1 - \alpha - \beta)$

### Why GARCH Matters for Risk
Under constant volatility, VaR is a fixed threshold. Under GARCH, VaR is **time-varying**:
it rises after turbulent periods and falls during calm ones — much more aligned with actual
market behaviour. The GARCH-based VaR also improves backtesting pass rates significantly.
""")
st.write("---")


# ── Data ──────────────────────────────────────────────────────────────
data = load_dax_index()
lr = get_log_returns(data)


# ── GARCH fitting ─────────────────────────────────────────────────────
@st.cache_data
def fit_garch11(returns):
    n = len(returns)
    var_unc = float(np.var(returns))

    def neg_loglik(params):
        omega, alpha_g, beta_g = float(params[0]), float(params[1]), float(params[2])
        if omega <= 0 or alpha_g <= 0 or beta_g <= 0 or alpha_g + beta_g >= 0.9999:
            return 1e10
        sigma2 = np.empty(n)
        sigma2[0] = var_unc
        for t in range(1, n):
            sigma2[t] = omega + alpha_g * returns[t - 1] ** 2 + beta_g * sigma2[t - 1]
            if sigma2[t] <= 0:
                return 1e10
        return 0.5 * float(np.sum(np.log(sigma2) + returns ** 2 / sigma2))

    best_val, best_params = np.inf, None
    for x0 in [
        [var_unc * 0.05, 0.08, 0.88],
        [var_unc * 0.10, 0.10, 0.85],
        [var_unc * 0.02, 0.05, 0.93],
    ]:
        res = minimize(
            neg_loglik, x0, method="L-BFGS-B",
            bounds=[(1e-9, None), (1e-6, 0.5), (1e-6, 0.999)],
            options={"ftol": 1e-12, "maxiter": 2000},
        )
        if res.fun < best_val:
            best_val, best_params = res.fun, res.x

    omega, alpha_g, beta_g = best_params
    sigma2 = np.empty(n)
    sigma2[0] = var_unc
    for t in range(1, n):
        sigma2[t] = omega + alpha_g * returns[t - 1] ** 2 + beta_g * sigma2[t - 1]

    return omega, alpha_g, beta_g, sigma2


with st.spinner("Fitting GARCH(1,1) via MLE — this runs once and is then cached..."):
    omega, alpha_g, beta_g, sigma2 = fit_garch11(lr)

var_unc = omega / (1 - alpha_g - beta_g)
persistence = alpha_g + beta_g

# ── Parameter display ─────────────────────────────────────────────────
st.header("1. Estimated Parameters")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("ω (variance floor)", f"{omega:.2e}")
c2.metric("α (ARCH effect)", f"{alpha_g:.4f}")
c3.metric("β (GARCH effect)", f"{beta_g:.4f}")
c4.metric("α + β (persistence)", f"{persistence:.4f}")
c5.metric("σ²∞ (unconditional var)", f"{var_unc:.2e}")

st.markdown(f"""
**Interpretation:** Persistence $\\alpha + \\beta = {persistence:.4f}$ is close to 1,
meaning volatility shocks take a long time to decay — consistent with the slow mean-reversion
observed in financial markets. The half-life of a volatility shock is approximately
$\\ln(0.5) / \\ln({persistence:.4f}) \\approx {np.log(0.5)/np.log(persistence):.0f}$ trading days.
""")
st.write("---")


# ── Conditional volatility ────────────────────────────────────────────
st.header("2. Conditional Volatility")
st.caption(
    "The annualised conditional volatility σ_t (scaled by √252) versus the constant "
    "historical volatility. Crisis periods are clearly visible as volatility spikes."
)

sigma_ann = np.sqrt(sigma2 * 252) * 100
const_vol = np.std(lr) * np.sqrt(252) * 100

fig1 = go.Figure()
fig1.add_hline(y=const_vol, line_dash="dot", line_color="grey",
               annotation_text=f"Constant vol: {const_vol:.1f}%", annotation_position="right")
fig1.add_trace(go.Scatter(
    x=np.arange(len(lr)), y=sigma_ann,
    mode="lines", name="GARCH σ_t (annualised %)", line=dict(color="crimson", width=1.5)
))
fig1.update_layout(
    xaxis_title="Trading Day", yaxis_title="Volatility (% p.a.)",
    xaxis=dict(showgrid=True), yaxis=dict(showgrid=True)
)
st.plotly_chart(fig1, use_container_width=True)
st.write("---")


# ── Standardised residuals ────────────────────────────────────────────
st.header("3. Standardised Residuals")
st.markdown(r"""
If the GARCH model is correctly specified, the **standardised residuals**
$\hat{\varepsilon}_t = X_t / \hat{\sigma}_t$ should be i.i.d. $\mathcal{N}(0,1)$.
We check this with a histogram and a QQ plot.
""")

std_resid = lr / np.sqrt(sigma2)

col1, col2 = st.columns(2)

with col1:
    x_range = np.linspace(-5, 5, 300)
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(
        x=std_resid, nbinsx=80, histnorm="probability density",
        name="Standardised residuals", marker_color="steelblue", opacity=0.7
    ))
    fig2.add_trace(go.Scatter(
        x=x_range, y=stats.norm.pdf(x_range),
        mode="lines", name="N(0,1)", line=dict(color="crimson", width=2)
    ))
    fig2.update_layout(xaxis_title="ε̂_t", yaxis_title="Density", title="Residual Distribution")
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    sorted_resid = np.sort(std_resid)
    n_r = len(sorted_resid)
    probs = (np.arange(1, n_r + 1) - 0.5) / n_r
    theoretical_q = stats.norm.ppf(probs)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=theoretical_q, y=sorted_resid, mode="markers",
        marker=dict(color="steelblue", size=2, opacity=0.5), name="Residuals"
    ))
    lim = max(abs(theoretical_q).max(), abs(sorted_resid).max())
    fig3.add_trace(go.Scatter(
        x=[-lim, lim], y=[-lim, lim], mode="lines",
        line=dict(color="crimson", dash="dash", width=2), name="45° line"
    ))
    fig3.update_layout(
        xaxis_title="N(0,1) quantiles", yaxis_title="Residual quantiles", title="QQ Plot"
    )
    st.plotly_chart(fig3, use_container_width=True)

skew_r = stats.skew(std_resid)
kurt_r = stats.kurtosis(std_resid)
_, p_jb = stats.jarque_bera(std_resid)
st.markdown(f"""
Residual skewness: **{skew_r:.3f}** | Excess kurtosis: **{kurt_r:.3f}** |
Jarque-Bera p-value: **{p_jb:.4f}**

{"✅ Residuals are close to normal — GARCH model fits well." if p_jb > 0.05 else
 "⚠️ Residuals still show non-normality. A GARCH model with Student-t innovations "
 "would improve the fit by allowing for remaining excess kurtosis."}
""")
st.write("---")


# ── GARCH-based VaR ──────────────────────────────────────────────────
st.header("4. GARCH-Based VaR")
st.markdown(r"""
Under GARCH(1,1), the **one-step-ahead conditional VaR** is:

$$\text{VaR}_\alpha^{\text{GARCH}} = \hat{\sigma}_{t+1} \cdot \Phi^{-1}(\alpha)$$

where $\hat{\sigma}_{t+1}^2 = \hat{\omega} + \hat{\alpha} X_t^2 + \hat{\beta} \hat{\sigma}_t^2$
is the **forecast** conditional variance. This gives a **time-varying VaR** that rises after
large shocks and contracts during quiet periods — directly addressing the clustering problem.
""")

alpha_var = st.select_slider("Confidence level α:", options=[0.90, 0.95, 0.975, 0.99], value=0.99)

var_garch = np.sqrt(sigma2) * stats.norm.ppf(alpha_var)
losses_idx = -lr

exceedances_g = np.where(losses_idx > var_garch)[0]
n_exc_g = len(exceedances_g)
exc_rate_g = n_exc_g / len(losses_idx) * 100

m1, m2, m3 = st.columns(3)
m1.metric("GARCH VaR exceedances", n_exc_g)
m2.metric("Exceedance rate", f"{exc_rate_g:.2f}%")
m3.metric("Expected rate", f"{(1 - alpha_var) * 100:.1f}%")

fig4 = go.Figure()
fig4.add_trace(go.Scatter(
    x=np.arange(len(lr)), y=losses_idx,
    mode="lines", name="Daily Loss", line=dict(color="steelblue", width=1), opacity=0.6
))
fig4.add_trace(go.Scatter(
    x=np.arange(len(lr)), y=var_garch,
    mode="lines", name=f"GARCH VaR_{alpha_var}", line=dict(color="orange", width=2)
))
fig4.add_trace(go.Scatter(
    x=exceedances_g, y=losses_idx[exceedances_g],
    mode="markers", name="Exceedance", marker=dict(color="red", size=5)
))
fig4.update_layout(
    xaxis_title="Trading Day", yaxis_title="Loss (log-return)",
    legend=dict(x=0, y=1), xaxis=dict(showgrid=True), yaxis=dict(showgrid=True)
)
st.plotly_chart(fig4, use_container_width=True)
st.write("---")


# ── Monte Carlo VaR (multi-day) ───────────────────────────────────────
st.header("5. Monte Carlo VaR — Multi-Day Horizon")
st.markdown(r"""
For a **10-day horizon**, analytical GARCH-based VaR is not available in closed form
because the variance process is non-linear. The solution is **Monte Carlo simulation**:

1. Fix the current state $(\hat{\sigma}_T^2, X_T)$ (last observation)
2. Simulate $B$ trajectories of length $h$ using the GARCH recursion with i.i.d. $\mathcal{N}(0,1)$ innovations
3. The $h$-day cumulative loss for each path is $-\sum_{t=1}^h X_t^{(b)}$
4. The Monte Carlo VaR is the empirical $\alpha$-quantile of the $B$ cumulative losses
""")

col1, col2 = st.columns(2)
with col1:
    horizon = st.slider("Forecast horizon h (days):", 1, 20, 10)
with col2:
    B_mc = st.select_slider("Monte Carlo paths B:", options=[5000, 10000, 50000], value=10000)


@st.cache_data
def garch_mc_var(omega, alpha_g, beta_g, last_sigma2, last_return, horizon, B, alpha_level, seed=42):
    rng = np.random.default_rng(seed)
    sigma2_sim = np.full(B, float(last_sigma2))
    x_sim = np.full(B, float(last_return))
    cum_loss = np.zeros(B)
    eps = rng.standard_normal((B, horizon))
    for t in range(horizon):
        sigma2_sim = omega + alpha_g * x_sim ** 2 + beta_g * sigma2_sim
        x_sim = np.sqrt(sigma2_sim) * eps[:, t]
        cum_loss -= x_sim
    return cum_loss, np.percentile(cum_loss, alpha_level * 100)


mc_losses, mc_var = garch_mc_var(
    omega, alpha_g, beta_g,
    last_sigma2=sigma2[-1], last_return=lr[-1],
    horizon=horizon, B=B_mc, alpha_level=alpha_var
)

mc_es = np.mean(mc_losses[mc_losses > mc_var])

col1, col2, col3 = st.columns(3)
col1.metric(f"{horizon}-day MC VaR_{alpha_var}", f"{mc_var:.5f}")
col2.metric(f"{horizon}-day MC ES_{alpha_var}", f"{mc_es:.5f}")
col3.metric("Current σ_t (daily)", f"{np.sqrt(sigma2[-1])*100:.3f}%")

fig5 = go.Figure()
x_hist = np.linspace(np.percentile(mc_losses, 0.1), np.percentile(mc_losses, 99.9), 200)
counts, bins = np.histogram(mc_losses, bins=80)
fig5.add_trace(go.Bar(
    x=(bins[:-1] + bins[1:]) / 2, y=counts,
    marker_color="steelblue", opacity=0.7, name=f"{horizon}-day cumulative loss"
))
fig5.add_vline(x=mc_var, line_dash="dash", line_color="orange", line_width=2,
               annotation_text=f"VaR = {mc_var:.4f}", annotation_position="top right")
fig5.add_vline(x=mc_es, line_dash="dash", line_color="crimson", line_width=2,
               annotation_text=f"ES = {mc_es:.4f}", annotation_position="top left")
fig5.update_layout(xaxis_title=f"{horizon}-day Cumulative Loss", yaxis_title="Frequency")
st.plotly_chart(fig5, use_container_width=True)

st.markdown(f"""
**Key insight:** The {horizon}-day GARCH Monte Carlo VaR ({mc_var:.4f}) is significantly
**larger** than $\\sqrt{{{horizon}}} \\times$ the 1-day VaR
({np.sqrt(horizon) * float(np.sqrt(sigma2[-1]) * stats.norm.ppf(alpha_var)):.4f} under the square-root-of-time rule).
This is because the GARCH variance process **amplifies** multi-day risk during high-volatility
regimes — the square-root-of-time rule only holds under i.i.d. returns.
""")
