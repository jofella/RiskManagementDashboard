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

with st.expander("📖 Intuition: GARCH in plain language"):
    st.markdown(r"""
    Imagine you're forecasting tomorrow's weather. A naive model says: "The average temperature
    in Berlin in January is 3°C — so tomorrow it will be 3°C." A better model says: "Yesterday
    was -8°C, so tomorrow is probably still cold — below the long-run average."

    GARCH does exactly this for volatility. The recursion
    $\sigma_t^2 = \omega + \alpha X_{t-1}^2 + \beta \sigma_{t-1}^2$ says:

    - **$\omega$:** The long-run "baseline" variance that volatility always gravitates towards.
    - **$\alpha X_{t-1}^2$:** Yesterday's shock. A large move yesterday (positive or negative)
      increases today's variance — the **ARCH effect**.
    - **$\beta \sigma_{t-1}^2$:** Yesterday's variance. If it was already high, today's will
      also be elevated — **variance momentum**, the **GARCH effect**.

    The sum $\alpha + \beta$ controls how quickly volatility mean-reverts. For DAX data,
    $\alpha + \beta \approx 0.97$–$0.99$ — very close to 1, meaning shocks take weeks or months
    to decay. This is called **integrated GARCH behaviour** and is a universal feature of
    equity markets.
    """)

with st.expander("📖 What does 'conditional' mean in GARCH?"):
    st.markdown(r"""
    The word **conditional** refers to conditioning on the information available at time $t-1$
    (the filtration $\mathcal{F}_{t-1}$).

    **Unconditional variance** $\text{Var}(X_t) = \sigma^2_\infty = \omega/(1-\alpha-\beta)$:
    the long-run average variance, the same number every day.

    **Conditional variance** $\text{Var}(X_t \mid \mathcal{F}_{t-1}) = \sigma_t^2$:
    the variance of tomorrow's return *given everything we know today* — it changes every day.

    The conditional variance is what actually matters for risk management. If markets were calm
    all week, the conditional 99% VaR for tomorrow is low. If there was a crash yesterday,
    the conditional VaR for today is much higher — and you should manage your position accordingly.

    **The analogy:** The unconditional probability of rain in London in July is 40%.
    But if it rained heavily yesterday and the forecast shows a depression moving in,
    the *conditional* probability for tomorrow might be 85%. The conditional estimate is
    more useful — and GARCH is the machine that computes the conditional volatility forecast.
    """)

with st.expander("📖 The ARCH test — is GARCH actually needed?"):
    st.markdown(r"""
    Before fitting a GARCH model, it's good practice to test whether time-varying volatility
    is actually present in the data. **Engle's ARCH test** does this by regressing squared
    residuals on their own lags:

    $$X_t^2 = a_0 + a_1 X_{t-1}^2 + \cdots + a_q X_{t-q}^2 + \varepsilon_t$$

    Under the null $H_0: a_1 = \cdots = a_q = 0$ (no ARCH effects), the test statistic
    $n \cdot R^2 \sim \chi^2_q$. A significant result means squared returns are autocorrelated
    — direct evidence of volatility clustering — and GARCH modelling is warranted.

    For essentially all major equity indices, the ARCH test rejects $H_0$ with extremely high
    confidence. The DAX is no exception: volatility clustering is one of the most robustly
    documented features of financial data, first described by Mandelbrot (1963) and formalised
    by Engle's ARCH model (1982, Nobel Prize 2003).
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
**Parameter interpretation:**

The **ARCH coefficient** $\\hat{{\\alpha}} = {alpha_g:.4f}$ measures the immediate impact of a
shock: a 1% daily return today increases tomorrow's variance by $\\hat{{\\alpha}} \\times (0.01)^2$,
which is a modest but non-negligible contribution to total variance.

The **GARCH coefficient** $\\hat{{\\beta}} = {beta_g:.4f}$ is the dominant driver:
it carries {beta_g/persistence*100:.0f}% of the total persistence. High $\\hat{{\\beta}}$ means
yesterday's *estimated variance* is very informative about today's — the model remembers
volatility states for a long time.

The **persistence** $\\hat{{\\alpha}} + \\hat{{\\beta}} = {persistence:.4f}$ governs the speed
of mean-reversion. The implied **half-life** of a volatility shock is
$\\ln(0.5)/\\ln({persistence:.4f}) \\approx {np.log(0.5)/np.log(persistence):.0f}$ trading days
($\\approx {np.log(0.5)/np.log(persistence)/21:.1f}$ months). This is consistent with
the empirical literature: equity volatility shocks typically take 1–3 months to fully dissipate.
Values of $\\alpha + \\beta$ near 1.0 (so-called **near-IGARCH** behaviour) are universal
across major equity indices and are sometimes interpreted as evidence that the unconditional
variance is non-stationary or very slowly varying.

The **unconditional variance** $\\hat{{\\sigma}}^2_\\infty = \\hat{{\\omega}}/(1 - \\hat{{\\alpha}} - \\hat{{\\beta}}) = {var_unc:.2e}$
corresponds to an annualised volatility of
$\\sqrt{{252 \\times {var_unc:.2e}}} \\times 100 \\approx {np.sqrt(252*var_unc)*100:.1f}\\%$,
consistent with the long-run average DAX volatility of approximately 20–22% p.a.
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
st.markdown(f"""
**Reading the conditional volatility series:**

The GARCH $\\hat{{\\sigma}}_t$ (annualised, %) ranges from approximately
{sigma_ann.min():.1f}% to {sigma_ann.max():.1f}% over the sample — a factor of
{sigma_ann.max()/sigma_ann.min():.1f}× between the calmest and most turbulent periods.
The constant historical volatility (dotted line, {const_vol:.1f}% p.a.) sits roughly in the middle
but severely misrepresents risk at both extremes.

Three crisis episodes are immediately identifiable as volatility spikes:
- **2001–2002 (dot-com bust / 9/11):** Sustained elevated volatility over ~18 months
- **2008–2009 (Global Financial Crisis):** The largest spike, with $\\hat{{\\sigma}}_t$ reaching
  its sample maximum. At peak, the GARCH 99% VaR was roughly {sigma_ann.max()/const_vol:.1f}×
  the constant-volatility VaR — capital requirements would differ by the same factor.
- **2020 (COVID-19):** A sharp but short-lived spike followed by rapid recovery, consistent
  with the $\\hat{{\\beta}} \\approx {beta_g:.2f}$ persistence decaying over weeks.

**Capital implications:** Under Basel II, banks using constant-volatility VaR would have
held the same amount of capital in January 2008 as in September 2008. The GARCH model
would have progressively increased the VaR estimate — and thus required capital — as the
crisis unfolded, providing a more forward-looking capital buffer.
""")
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

_skew_assess = "Acceptable" if abs(skew_r) < 0.3 else "Non-trivial asymmetry remaining"
_kurt_assess = "Substantial — Student-t innovations recommended" if kurt_r > 1 else "Mild — Gaussian innovations adequate"
_jb_assess = "Not rejected" if p_jb > 0.05 else "Rejected — residuals not normal"

if p_jb > 0.05:
    _model_assessment = (
        "**Model assessment — Gaussian innovations adequate:** The GARCH(1,1) with normal "
        "innovations captures the bulk of the temporal dependence in variance. Residual excess "
        "kurtosis is low, and the JB test does not reject normality at conventional significance levels."
    )
else:
    _implied_nu = f"{4 + 6 / max(kurt_r, 0.1):.0f}"
    _model_assessment = (
        f"**Model assessment — Student-t innovations recommended:** Even after GARCH filtering, the\n"
        f"standardised residuals exhibit excess kurtosis of {kurt_r:.2f}. Under i.i.d. normality the\n"
        f"expected excess kurtosis is 0; the observed value suggests **remaining fat-tail structure**\n"
        f"that the conditional variance model does not fully explain.\n\n"
        f"A **GARCH(1,1) with Student-t innovations** (estimating $\\\\nu$ jointly with "
        f"$\\\\omega, \\\\alpha, \\\\beta$) would model this residual kurtosis explicitly: "
        f"$\\\\hat{{\\\\varepsilon}}_t \\\\sim t_\\\\nu(0,1)$ with "
        f"$\\\\hat{{\\\\nu}} \\\\approx 4 + 6/{kurt_r:.2f} \\\\approx {_implied_nu}$ degrees of freedom "
        f"implied by the residual kurtosis alone. The Jarque-Bera rejection confirms this is statistically "
        f"significant, not a small-sample artefact."
    )

st.markdown(f"""
**Residual diagnostics:**

| Statistic | Value | Benchmark (i.i.d. Normal) | Assessment |
|---|---|---|---|
| Skewness | {skew_r:.4f} | 0 | {_skew_assess} |
| Excess kurtosis | {kurt_r:.4f} | 0 | {_kurt_assess} |
| Jarque-Bera p-value | {p_jb:.6f} | > 0.05 under H₀ | {_jb_assess} |

{_model_assessment}

**Interpretation of the QQ plot:** Deviations from the 45° line in the tails of the standardised
residuals confirm that even after removing conditional heteroscedasticity, the innovations
retain some fat-tail structure. This motivates either (a) Student-t innovations in the GARCH
model, or (b) applying EVT-based methods to the GARCH residuals rather than the raw returns.
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
st.markdown(f"""
**GARCH VaR performance:** The observed exceedance rate of {exc_rate_g:.3f}% compares against
the expected rate of {(1-alpha_var)*100:.1f}% under correct model specification.

A key observable property of GARCH-based VaR is that **exceedances are not clustered in
time**. Unlike a constant-volatility VaR (where exceedances concentrate in crisis periods),
the GARCH model raises its VaR estimate *as* volatility increases, so exceedances remain
approximately uniformly distributed over time. This is precisely what the Kupiec test and the
Christoffersen independence test measure in the backtesting chapter.

Visually, notice how the orange VaR line expands dramatically during crisis periods (2008, 2020)
and contracts during calm periods — tracking the realised loss distribution in real time.
This time-varying property is the fundamental advantage of GARCH over constant-volatility models
for regulatory capital calculations.
""")
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
