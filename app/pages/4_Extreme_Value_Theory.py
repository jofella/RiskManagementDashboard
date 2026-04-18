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

with st.expander("📖 Intuition: Why do we need EVT at all?"):
    st.markdown(r"""
    Consider estimating the 99.9% VaR from 10 years of daily data (~2,500 observations).
    You need to estimate the **2.5th worst loss** — but you have only 2–3 observations in that
    extreme quantile region. Any direct estimate is dominated by noise.

    **The normal distribution "solves" this** by extrapolating from the centre (mean and variance)
    to the tail using the Gaussian formula. But as we've seen, this extrapolation is badly wrong —
    it dramatically underestimates tail probabilities.

    **EVT solves this correctly.** It says: *"I don't know what the full distribution looks like
    in the centre, but I can prove mathematically that the tail — above a high threshold —
    must converge to a GPD, regardless of what distribution generated the data."*

    This is the power of EVT: it provides a **universally valid model for extreme quantiles**
    without requiring you to specify the full distribution. The only question is:
    which shape parameter $\xi$ does the tail have?

    **Analogy:** The Central Limit Theorem tells you that sums of random variables converge to
    a normal, regardless of the original distribution. EVT is the analogous theorem for *maxima*
    and *exceedances* — not averages.
    """)

with st.expander("📖 Intuition: What is a GPD and what does ξ mean?"):
    st.markdown(r"""
    The **Generalised Pareto Distribution** has CDF:

    $$G_{\xi,\sigma}(y) = 1 - \left(1 + \frac{\xi y}{\sigma}\right)^{-1/\xi}, \quad y > 0$$

    The shape parameter $\xi$ controls tail behaviour:

    | $\xi$ | Tail type | Example distributions | Financial relevance |
    |---|---|---|---|
    | $\xi > 0$ | **Heavy (Pareto-type)** polynomial decay | Pareto, Student-t, log-Cauchy | Equities, credit losses — typical $\xi \approx 0.3$ |
    | $\xi = 0$ | **Light (exponential)** | Normal, lognormal, exponential | Thin-tailed assets |
    | $\xi < 0$ | **Bounded** (finite upper limit) | Beta, uniform | Rare in finance |

    **Intuition for $\xi > 0$:** The survival function $P(X > x) \sim x^{-1/\xi}$ decays as a
    power law — polynomially, not exponentially. This means extreme quantiles grow *much faster*
    than the normal prediction as $\alpha \to 1$. Doubling $\alpha$ from 99% to 99.5% does not
    just double the VaR — for $\xi = 0.3$, it roughly increases it by a factor of $2^{\xi} \approx 1.23$.

    **In practice for DAX returns:** Typical estimates are $\hat{\xi} \approx 0.2$–$0.4$,
    confirming heavy tails and validating the use of EVT over normal-based extrapolation.
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
st.markdown(r"""
A **Quantile-Quantile (QQ) plot** compares the empirical quantiles of the data to the theoretical
quantiles of a reference distribution. If the data follows the reference distribution, the points
lie on a straight line. **Deviations in the upper-right tail** indicate **heavier tails** than
the reference — the empirical extreme quantiles are larger than the model predicts.

The QQ plot is a simple but powerful diagnostic: it reveals in one glance whether the normal
assumption is tenable for the tail region where VaR and ES estimates actually matter.
""")

with st.expander("📖 How to read a QQ plot"):
    st.markdown(r"""
    The QQ plot places **theoretical quantiles** on the x-axis and **empirical quantiles** on
    the y-axis. Both are sorted from smallest to largest.

    - **Points on the 45° line:** The distribution fits perfectly in that region.
    - **Points curving upward (above the line) in the right tail:** The empirical tail is
      *heavier* than the reference — actual extreme losses are larger than predicted.
    - **Points curving downward (below the line) in the left tail:** Negative returns are
      also more extreme than predicted (this is what you see for losses in the lower-left corner).

    **What to look for here:** With a normal reference, you'll see the characteristic S-shape
    or hockey-stick deviation in both tails — evidence of leptokurtosis. As you switch to
    Student-t with $\nu = 4$ or $\nu = 6$, the fit in the tails improves, confirming that
    financial returns have approximately Student-t tails.
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

st.markdown(r"""
**Findings:** Under the normal reference, the characteristic **S-shape** is immediately visible:
empirical quantiles lie *above* the reference line in the upper-right tail (losses are more
extreme than the normal predicts) and *below* in the lower-left (gains are also more extreme).
This simultaneous deviation in both tails is the signature of **leptokurtosis** — the
distribution is more peaked at the centre and heavier in the tails than the normal.

Switching to **Student-t ($\nu = 4$ or $\nu = 6$)** substantially improves the tail fit:
the S-curve flattens because the t-distribution's power-law tails better match the
empirical extremes. However, even the t-fit is imperfect in the very extreme quantiles —
consistent with the finding from the histogram (Section 1 in Explore Data) that a single
parametric family cannot perfectly describe both the centre and the extreme tail.

This visual evidence motivates EVT: rather than forcing a global parametric fit, EVT focuses
exclusively on the tail region above a high threshold, where the GPD approximation is valid.
""")
st.write("---")


# ============================================================
# SECTION 2: Hill Estimator
# ============================================================
st.header("2. Regular Variation and Heavy Tails")
st.markdown(r"""
The Hill estimator and the POT method are grounded in the theory of **regularly varying functions**
— the precise mathematical language for heavy-tailed distributions.

### Regular Variation
A positive measurable function $f$ is **regularly varying** at $\infty$ with index $\rho \in \mathbb{R}$
if for all $t > 0$:

$$\lim_{x \to \infty} \frac{f(tx)}{f(x)} = t^\rho$$

Written $f \in RV_\rho$. The case $\rho = 0$ is called **slowly varying** (e.g. $\log x$, constants).

A random variable $X$ (or its distribution) is **regularly varying** with tail index $\alpha > 0$ if:

$$P(X > x) = x^{-\alpha} L(x), \quad x \to \infty$$

where $L$ is a slowly varying function. Equivalently, $P(X > \cdot) \in RV_{-\alpha}$.

### Key Examples
| Distribution | Survival function | Regularly varying? | Tail index $\alpha$ |
|---|---|---|---|
| Pareto($\alpha$) | $(x/x_m)^{-\alpha}$ | ✅ Yes | $\alpha$ |
| Student-t($\nu$) | $\sim c \cdot x^{-\nu}$ | ✅ Yes | $\nu$ |
| Log-Cauchy | $\sim (\log x)^{-1}$ (slowly varying) | ✅ Yes (borderline) | $0$ |
| **Exponential** | $e^{-\lambda x}$ (decays faster than any power) | ❌ **No** | — |
| Normal | $e^{-x^2/2}$ (super-exponential) | ❌ **No** | — |

**The exponential distribution is NOT regularly varying** — its survival function $e^{-\lambda x}$
decays exponentially, not polynomially, so the limit $e^{-\lambda(tx)}/e^{-\lambda x} = e^{-\lambda(t-1)x} \to 0$
(not $t^\rho$) for any fixed $t > 1$. This is why EVT tail extrapolation is only valid for
Pareto-type tails, not exponential or normal tails.

### Why Regular Variation Matters for Risk
- **Finite moments:** A regularly varying variable $X$ with index $\alpha$ has $E[X^p] < \infty$
  iff $p < \alpha$. For $\alpha = 3$ (typical equity), mean and variance exist but kurtosis may not.
- **VaR and ES extrapolation:** Under regular variation, high quantiles obey
  $\text{VaR}_\alpha(L) \approx c \cdot (1-\alpha)^{-1/\xi}$ for large $\alpha$ — so the Hill
  estimator of $\xi$ directly enables tail extrapolation beyond the data.
""")

with st.expander("📖 Estimating VaR and ES under regular variation"):
    st.markdown(r"""
    Suppose $P(L > x) \approx C x^{-1/\xi}$ for large $x$ (Pareto-type tail with index $1/\xi$).
    Given the Hill estimate $\hat{\xi}$ at threshold $k$ (using the $k$-th largest loss $X_{(n-k)}$):

    **Extremal CDF estimate:**
    $$\hat{P}(L > x) = \frac{k}{n} \left(\frac{X_{(n-k)}}{x}\right)^{1/\hat{\xi}}, \quad x > X_{(n-k)}$$

    **VaR extrapolation** (inverting the above for target level $\alpha > 1 - k/n$):
    $$\widehat{\text{VaR}}_\alpha = X_{(n-k)} \cdot \left(\frac{k/n}{1-\alpha}\right)^{\hat{\xi}}$$

    **ES extrapolation** (using the integral representation):
    $$\widehat{\text{ES}}_\alpha = \frac{\widehat{\text{VaR}}_\alpha}{1 - \hat{\xi}}, \quad \hat{\xi} < 1$$

    The ES formula shows that for $\hat{\xi}$ close to 1, ES becomes very large — meaning the
    expected loss in the tail is many multiples of VaR. For $\hat{\xi} \geq 1$, the mean excess
    is infinite and ES is formally undefined.

    **Comparison with GPD/POT:** The POT method gives a more principled estimate using all
    exceedances, while the Hill approach uses only the ratio of order statistics. For moderate
    sample sizes the POT method is generally preferred.
    """)

st.write("---")


st.header("3. Hill Estimator: Tail Index")
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

st.markdown(fr"""
**Reading the Hill plot:** Reliability requires the estimator to form a **stable plateau** —
a region of $k$ values where $\hat{{\xi}}_k$ fluctuates around a constant level before
either rising (upward bias from including non-tail observations) or falling (downward bias
from too few order statistics). At $k = {k_ref}$, $\hat{{\xi}} \approx {xi_at_k:.3f}$.

**Benchmark comparison:** For major equity indices, the EVT literature consistently estimates
$\xi \in (0.2, 0.4)$, corresponding to Student-t tail behaviour with effective degrees of
freedom $\hat{{\nu}} = 1/\hat{{\xi}} \approx {1/xi_at_k:.1f}$. The current estimate is
{"within" if 0.2 <= xi_at_k <= 0.4 else "outside"} this benchmark range.

**Moment conditions:** A regularly varying distribution with index $1/\hat{{\xi}}$ has
finite moments up to order $p < 1/\hat{{\xi}} \approx {1/xi_at_k:.1f}$. In particular:
- Finite mean requires $\xi < 1$ ✅ (always satisfied here)
- Finite variance requires $\xi < 0.5$ {"✅" if xi_at_k < 0.5 else "❌"} ($\hat{{\xi}} = {xi_at_k:.3f}$)
- Finite kurtosis requires $\xi < 0.25$ {"✅" if xi_at_k < 0.25 else "❌"} — this is where
  the normal-based kurtosis estimate breaks down if $\xi \geq 0.25$.

**Capital implications:** Under regular variation, the ES formula gives
$\widehat{{\text{{ES}}}}_\alpha = \widehat{{\text{{VaR}}}}_\alpha / (1 - \hat{{\xi}})$.
With $\hat{{\xi}} \approx {xi_at_k:.3f}$, ES exceeds VaR by a factor of
$1/(1-{xi_at_k:.3f}) \approx {1/(1-xi_at_k):.2f}$ — a substantial tail severity premium
that normal-based models miss entirely.
""")
st.write("---")


# ============================================================
# SECTION 3: Mean Excess Plot (MEP)
# ============================================================
st.header("4. Mean Excess Plot (MEP)")
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

st.markdown(r"""
**Reading the plot — what the slope tells you:** For a GPD-distributed excess distribution,
$e(u) = (\sigma + \xi u)/(1-\xi)$, so the slope of the linear section equals $\xi/(1-\xi)$.
A **positive slope** therefore directly confirms $\xi > 0$ — a Pareto-type heavy tail.
A slope near zero would indicate an exponential tail ($\xi = 0$), and a downward slope
would indicate a bounded tail ($\xi < 0$, rare in financial returns).

**Threshold selection:** The optimal threshold $u^*$ is the lowest value of $u$ from which
$e(u)$ is approximately linear (upward). Choosing $u^*$ too low includes non-tail observations
that corrupt the GPD fit; too high leaves too few exceedances for reliable MLE estimation.

**The right tail vs the turn-down:** It is normal for the plot to become erratic or turn
downward at very high thresholds — when only 5–10 observations remain above $u$, the
sample mean excess is dominated by noise. Disregard the rightmost portion and focus on
the stable linear region in the middle of the plot.

**Bias-variance tradeoff:** This threshold choice is the central estimation challenge of
the POT method — the same bias-variance tradeoff as in kernel density estimation.
Automated threshold selection methods (e.g. Wadsworth or Bader-Yan) formalise this.
""")
st.write("---")


# ============================================================
# SECTION 4: POT Method — GPD Fitting
# ============================================================
st.header("5. Peak-over-Threshold (POT) Method")
st.markdown(r"""
For losses exceeding a high threshold $u$, the **Pickands–Balkema–de Haan theorem** states that
the excess distribution converges to a **Generalized Pareto Distribution (GPD)**:

$$F_u(y) = P(L - u \leq y \mid L > u) \approx G_{\xi, \sigma}(y) = 1 - \left(1 + \frac{\xi y}{\sigma}\right)^{-1/\xi}$$

where $\xi$ is the **shape** (tail index) and $\sigma > 0$ is the **scale** parameter.
The GPD parameters are estimated from the exceedances by **Maximum Likelihood Estimation**.
Once fitted, the tail CDF, VaR, and ES can be extrapolated to any confidence level.
""")

with st.expander("📖 Maximum Likelihood Estimation — how it works and why it's needed"):
    st.markdown(r"""
    **The principle:** Given $n$ observations $x_1, \ldots, x_n$ assumed i.i.d. from a
    parametric family $f_\theta$, the **MLE** finds the parameter $\hat{\theta}$ that makes
    the observed data most probable:

    $$\hat{\theta}_{\text{MLE}} = \arg\max_\theta \sum_{i=1}^n \log f_\theta(x_i)$$

    (We maximise the **log-likelihood** for numerical convenience — log turns products into sums.)

    **Why is MLE needed in risk management?**
    - Historical simulation only works if you have enough data in the tail. For 99.9% VaR,
      you need thousands of observations — rarely available.
    - MLE fits a parametric model (GPD, GARCH, t-copula) to whatever data exists, then
      extrapolates analytically to confidence levels beyond the sample.

    **MLE for the GPD** (POT method): Given exceedances $y_1, \ldots, y_{n_u}$ above threshold $u$,
    the log-likelihood is:
    $$\ell(\xi, \sigma) = -n_u \log\sigma - \left(1 + \frac{1}{\xi}\right)\sum_{i=1}^{n_u} \log\left(1 + \frac{\xi y_i}{\sigma}\right)$$
    Maximising over $(\xi, \sigma)$ gives $(\hat{\xi}, \hat{\sigma})$.

    **For dependent data (GARCH):** The log-likelihood is no longer a sum of i.i.d. terms.
    Instead it decomposes as:
    $$\ell(\theta) = \sum_{t=1}^n \log f_\theta(x_t \mid x_1, \ldots, x_{t-1}) = \sum_{t=1}^n \log f_\theta(x_t \mid \sigma_t^2(\theta))$$
    where $\sigma_t^2$ is the conditional variance given all past observations. This is the
    **conditional log-likelihood** and is what the GARCH MLE page computes.
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

        st.markdown(fr"""
        **Reading the table — divergence by confidence level:** The gap between EVT and normal
        estimates is small near $\alpha = 0.95$ (both methods are still anchored in the body
        of the sample distribution), but grows substantially at $\alpha \geq 0.99$ and is most
        pronounced at $\alpha = 0.999$.

        **Why the divergence accelerates:** The normal quantile grows logarithmically as
        $\Phi^{{-1}}(\alpha) \sim \sqrt{{2\log(1/(1-\alpha))}}$, while the GPD quantile grows
        as $(1-\alpha)^{{-\xi}}$ — a power law. For $\hat{{\xi}} \approx {xi_fit:.2f}$, the
        ratio $\text{{VaR}}_\alpha^{{\text{{EVT}}}}/\text{{VaR}}_\alpha^{{\text{{Normal}}}}$
        grows without bound as $\alpha \to 1$. At $\alpha = 0.999$, this factor can easily
        reach 2–3× — meaning the normal model requires only one-third the capital that EVT
        prescribes for a 1-in-1000 scenario.

        **Regulatory context:** Solvency II requires the 99.5% one-year VaR; FRTB uses 97.5% ES.
        Both are in the region where the EVT–normal divergence is already material.
        The EVT estimate is therefore the appropriate reference for regulatory capital calculations.
        ES values are also substantially larger than VaR at all levels, reflecting the heavy-tail
        severity captured by $\hat{{\xi}} \approx {xi_fit:.2f} > 0$: $\text{{ES}}_\alpha /
        \text{{VaR}}_\alpha \approx 1/(1-\hat{{\xi}}) \approx {1/(1-xi_fit):.2f}$.
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

        st.markdown(r"""
        **Log-scale tail plot — what to look for:** On a log-scale, an **exponential** survival
        function $P(L>x) = e^{-\lambda x}$ appears as a straight line with negative slope.
        A **power-law** survival function $P(L>x) \sim x^{-1/\xi}$ appears as a straight line
        with slope $-1/\xi$ on a **log-log** scale — meaning it curves *upward* on the
        log-linear scale used here, decaying more slowly than exponential.

        **The three curves:** The empirical tail (blue dots) clearly lies *above* the normal
        (orange) in the extreme region — the data decays more slowly than the normal predicts.
        The GPD (red) follows the empirical tail closely, capturing the power-law decay.
        The normal curve collapses too rapidly because its super-exponential tails $e^{-x^2/2}$
        are incompatible with the polynomial decay that the data exhibit.

        **Practical consequence:** At $P(L>x) = 0.1\%$ (1-in-1000 day), the normal model
        places the threshold substantially *below* the GPD estimate — meaning normal-based
        models allocate insufficient capital for low-frequency, high-severity events.
        This is precisely the failure mode that EVT was designed to correct.
        """)

    except Exception as e:
        st.error(f"GPD fitting failed: {e}. Try adjusting the threshold.")
