import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy import stats
from scipy.stats import nbinom, pareto

from util.data_utils import load_dax_companies, get_log_returns

STOCK_NAMES = ["BMW", "SAP", "Volkswagen", "Continental", "Siemens"]
WINDOW = 252

st.title("🔗 Copulas & Dependence Structures")
st.markdown(r"""
In portfolio risk management, it is rarely sufficient to model each asset in isolation —
the **joint behaviour** of assets under stress is what determines whether a portfolio survives
an extreme event. The central challenge is that assets may show only moderate correlation under
normal conditions but move together violently during crises, a phenomenon known as
**tail dependence**. Copulas provide the mathematical language to describe this precisely.

### Sklar's Theorem: The Fundamental Decomposition
Every joint distribution $F$ with continuous marginals $F_1, \ldots, F_d$ can be
*uniquely* decomposed as:

$$F(x_1, \ldots, x_d) = C\!\left(F_1(x_1), \ldots, F_d(x_d)\right)$$

where $C: [0,1]^d \to [0,1]$ is a **copula** — a multivariate distribution on the unit
hypercube with uniform $\mathcal{U}(0,1)$ marginals (Sklar, 1959). Conversely, any copula $C$
combined with arbitrary marginals $F_i$ defines a valid joint distribution.

This decomposition is powerful because it **separates two orthogonal modelling choices**:
1. **Marginal distributions** $F_i$ — capturing the individual behaviour of each asset
   (e.g. fat-tailed Student-t or fitted EVT distributions).
2. **Copula $C$** — capturing the dependence structure independently of the marginals.

### Tail Dependence
The **upper tail dependence coefficient** measures the probability that one variable is extreme
*given* the other is extreme:

$$\lambda_u = \lim_{\alpha \to 1} P\!\left(F_1(X_1) > \alpha \mid F_2(X_2) > \alpha\right)$$

- $\lambda_u > 0$: **upper tail dependence** — extremes co-occur; critical for portfolio stress
- $\lambda_u = 0$: **asymptotic independence** — extremes become independent in the limit

The **Gaussian copula has $\lambda_u = 0$** regardless of correlation — a critical limitation
exposed during the 2008 financial crisis, where structured products priced under Gaussian copula
assumptions experienced simultaneous defaults far beyond model predictions.

### Copula Families Covered Here
| Copula | Tail dependence | Key parameter | Typical use |
|---|---|---|---|
| **Gaussian** | None ($\lambda_u = \lambda_l = 0$) | Correlation $R$ | Baseline linear dependence |
| **Normal Mixture** | Flexible (mixture) | $R_1, R_2, p$ | Regime-switching dependence |
| **Clayton** | Lower tail ($\lambda_l > 0$) | $\vartheta > 0$ | Simultaneous small losses |
| **Reverse Clayton** | Upper tail ($\lambda_u > 0$) | $\vartheta > 0$ | Simultaneous large losses |
| **t-Copula** | Both tails ($\lambda_u = \lambda_l > 0$) | $\nu, R$ | Heavy joint tail risk |
""")
st.write("---")


# ============================================================
# SECTION 1: Rolling Correlations
# ============================================================
st.header("1. Linear vs. Rank Correlation")
st.markdown("""
Two measures of dependence between two random variables $X_1, X_2$:

- **Linear (Pearson) correlation** $\\hat{\\varrho}_L = \\frac{\\widehat{\\text{Cov}}(X_1, X_2)}{\\hat{\\sigma}_1 \\hat{\\sigma}_2}$
  — sensitive to the magnitude of observations, dominated by outliers.

- **Spearman's rank correlation** $\\hat{\\varrho}_S$ — works with the *ranks* of observations,
  reducing outlier sensitivity. Defined as linear correlation of the rank vectors.

Both are estimated over a rolling 252-day window on BMW vs. Volkswagen log-returns.
""")

data = load_dax_companies()
lr = get_log_returns(data)
lr_BMW = lr[:, 0]
lr_VW = lr[:, 2]
T = len(lr_BMW)


@st.cache_data
def compute_rolling_correlations(lr1, lr2, window):
    n = len(lr1)
    rho_l = np.full(n, np.nan)
    rho_s = np.full(n, np.nan)

    def spearman(x, y):
        n = len(x)
        rank_x = np.argsort(np.argsort(x)) + 1
        rank_y = np.argsort(np.argsort(y)) + 1
        mean_diff = (n + 1) / 2
        scaling = 12 / (n * (n**2 - 1))
        return scaling * np.sum((rank_x - mean_diff) * (rank_y - mean_diff))

    for t in range(window, n):
        x, y = lr1[t - window:t], lr2[t - window:t]
        cov = np.cov(x, y)
        rho_l[t] = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        rho_s[t] = spearman(x, y)

    return rho_l, rho_s


rho_l, rho_s = compute_rolling_correlations(lr_BMW, lr_VW, WINDOW)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=np.arange(T), y=rho_l,
    mode="lines", name="Pearson ρ_L", line=dict(color="steelblue", width=1.5)
))
fig.add_trace(go.Scatter(
    x=np.arange(T), y=rho_s,
    mode="lines", name="Spearman ρ_S", line=dict(color="crimson", width=1.5, dash="dash")
))
fig.add_hline(y=0, line_dash="dot", line_color="grey", line_width=1)
fig.update_layout(
    xaxis_title="Trading Day", yaxis_title="Correlation",
    legend=dict(x=0, y=1), xaxis=dict(showgrid=True), yaxis=dict(showgrid=True, range=[-0.2, 1.1])
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
**Findings:** Both measures show strong positive correlation between BMW and VW throughout the
entire period — not surprising given both are automotive companies. Spearman's ρ_S closely tracks
linear ρ_L, suggesting the dependence is largely linear. This supports a **Gaussian copula** as
a reasonable model — its dependence structure is fully determined by linear correlation.
""")
st.write("---")


# ============================================================
# SECTION 2: Gaussian Copula
# ============================================================
st.header("2. Gaussian Copula")
st.markdown(r"""
The **bivariate Gaussian copula** $C_R^{\text{Ga}}$ with correlation matrix $R$ is sampled as follows:

1. Compute the Cholesky decomposition $R = A A^\top$
2. Draw $Z \sim \mathcal{N}(0, I_2)$ (independent standard normals)
3. Set $X = AZ$ (introduces correlation)
4. Transform: $U_i = \Phi(X_i)$ where $\Phi$ is the standard normal CDF

The resulting $(U_1, U_2) \in [0,1]^2$ has uniform marginals and correlation $\varrho_L$.
""")

col1, col2 = st.columns(2)
with col1:
    rho_input = st.slider("Correlation ρ:", min_value=-0.99, max_value=0.99, value=0.80, step=0.01)
with col2:
    n_samples = st.select_slider("Samples n:", options=[1000, 5000, 10000], value=5000)


def build_corr_matrix(rho):
    return np.array([[1.0, rho], [rho, 1.0]])


def sample_gauss_copula(n, R, seed=None):
    rng = np.random.default_rng(seed)
    A = np.linalg.cholesky(R)
    Z = rng.standard_normal((2, n))
    X = A @ Z
    return stats.norm.cdf(X)


R_pos = build_corr_matrix(rho_input)
R_neg = build_corr_matrix(-abs(rho_input))

s_pos = sample_gauss_copula(n_samples, R_pos, seed=42)
s_neg = sample_gauss_copula(n_samples, R_neg, seed=42)

fig2 = make_subplots(rows=1, cols=2, subplot_titles=(f"ρ = {rho_input}", f"ρ = {-abs(rho_input):.2f}"))
fig2.add_trace(go.Scatter(
    x=s_pos[0], y=s_pos[1], mode="markers", name=f"ρ = {rho_input}",
    marker=dict(color="steelblue", size=2, opacity=0.5)
), row=1, col=1)
fig2.add_trace(go.Scatter(
    x=s_neg[0], y=s_neg[1], mode="markers", name=f"ρ = {-abs(rho_input):.2f}",
    marker=dict(color="crimson", size=2, opacity=0.5)
), row=1, col=2)
fig2.update_xaxes(title_text="U₁", range=[0, 1])
fig2.update_yaxes(title_text="U₂", range=[0, 1])
st.plotly_chart(fig2, use_container_width=True)

st.markdown("""
**Note:** The Gaussian copula has **no tail dependence** ($\\lambda_u = \\lambda_l = 0$).
Extreme events in one variable become asymptotically independent of extremes in the other —
a known limitation for financial risk modelling.
""")
st.write("---")


# ============================================================
# SECTION 3: Normal Mixture Copula
# ============================================================
st.header("3. Normal Mixture Copula")
st.markdown(r"""
A **normal mixture copula** mixes two Gaussian copulas:

$$C_{R_1, R_2, p}^{\text{Mix}} := p \cdot C_{R_1}^{\text{Ga}} + (1-p) \cdot C_{R_2}^{\text{Ga}}$$

To sample: draw a Bernoulli($p$) indicator and sample from $C_{R_1}^{\text{Ga}}$ or
$C_{R_2}^{\text{Ga}}$ accordingly.

**Use case:** Model situations where asset magnitudes are correlated (large moves together)
but the sign of the correlation is uncertain — e.g. short-term vs. long-term dynamics.
""")

col1, col2, col3 = st.columns(3)
with col1:
    rho1_mix = st.slider("ρ₁ (component 1):", -0.99, 0.99, 0.80, 0.01, key="mix_rho1")
with col2:
    rho2_mix = st.slider("ρ₂ (component 2):", -0.99, 0.99, -0.80, 0.01, key="mix_rho2")
with col3:
    p_mix = st.slider("Mixing weight p:", 0.01, 0.99, 0.75, 0.01)


def sample_mix_copula(n, R1, R2, p, seed=None):
    rng = np.random.default_rng(seed)
    indicators = rng.binomial(1, p, n)
    samples = np.zeros((2, n))
    idx1 = np.where(indicators == 1)[0]
    idx0 = np.where(indicators == 0)[0]
    if len(idx1):
        s1 = sample_gauss_copula(len(idx1), R1, seed=seed)
        samples[:, idx1] = s1
    if len(idx0):
        s0 = sample_gauss_copula(len(idx0), R2, seed=seed)
        samples[:, idx0] = s0
    return samples


R1_mix = build_corr_matrix(rho1_mix)
R2_mix = build_corr_matrix(rho2_mix)
s_mix = sample_mix_copula(n_samples, R1_mix, R2_mix, p_mix, seed=42)

fig3 = make_subplots(
    rows=1, cols=3,
    subplot_titles=(f"C^Ga(ρ={rho1_mix})", f"C^Ga(ρ={rho2_mix})", f"Mixture (p={p_mix})")
)
for col_idx, (s, color) in enumerate([(s_pos, "steelblue"), (s_neg, "crimson"), (s_mix, "darkorchid")], 1):
    fig3.add_trace(go.Scatter(
        x=s[0], y=s[1], mode="markers",
        marker=dict(color=color, size=2, opacity=0.4),
        showlegend=False
    ), row=1, col=col_idx)
fig3.update_xaxes(title_text="U₁", range=[0, 1])
fig3.update_yaxes(title_text="U₂", range=[0, 1])
st.plotly_chart(fig3, use_container_width=True)

st.markdown("""
**Interpretation:** The mixture copula (right) combines the structures of both components.
With $p=0.75$ and $\\varrho_1=0.8$, $\\varrho_2=-0.8$, it captures both correlated and
anti-correlated regimes — something a single Gaussian copula cannot achieve.
""")
st.write("---")


# ============================================================
# SECTION 4: Clayton / Reverse Clayton Copula
# ============================================================
st.header("4. Clayton Copula: Insurance Application")
st.markdown(r"""
The **Clayton copula** with parameter $\vartheta > 0$ has **lower tail dependence**
$\lambda_\ell = 2^{-1/\vartheta}$ but no upper tail dependence ($\lambda_u = 0$).

The **reverse Clayton copula** $C_\vartheta^{\text{rCl}}$ flips this: $\tilde{U} = 1 - U$ has a
Clayton copula. It has upper tail dependence $\lambda_u = 2^{-1/\vartheta}$ and $\lambda_\ell = 0$.

**Sampling via the Marshall-Olkin method:**
1. Draw $V \sim \text{Gamma}(1/\vartheta, 1)$
2. Draw $\tilde{U}_1, \tilde{U}_2 \sim \text{Uniform}(0,1)$ independently
3. $\tilde{U}_i = \left(-\frac{\ln \tilde{U}_i}{V} + 1\right)^{-1/\vartheta}$ → Clayton copula
4. $U_i = 1 - \tilde{U}_i$ → Reverse Clayton copula

**Insurance application:** An insurer pays $X = N \cdot S$, where $N \sim \text{NB}(5, 0.01)$
(claim count) and $S \sim \text{Pareto}(5)$ (claim size). Upper tail dependence reflects:
*when claims are large, they also tend to be numerous*.
""")

col1, col2 = st.columns(2)
with col1:
    theta_clayton = st.slider("Clayton parameter ϑ:", min_value=0.5, max_value=20.0, value=10.0, step=0.5)
with col2:
    n_ins = st.select_slider("Simulations:", options=[5000, 10000, 50000], value=10000)


@st.cache_data
def simulate_insurance(theta, n, seed=20):
    rng = np.random.default_rng(seed)
    V = rng.gamma(shape=1 / theta, scale=1, size=n)
    raw = rng.uniform(size=(2, n))
    U_tilde = (-np.log(raw) / V + 1) ** (-1 / theta)
    U = 1 - U_tilde  # reverse Clayton

    N_dep = nbinom(n=5, p=0.01).ppf(U[0])
    S_dep = pareto(b=5).ppf(U[1])

    rng2 = np.random.default_rng(seed)
    N_ind = nbinom(n=5, p=0.01).rvs(size=n, random_state=rng2)
    S_ind = pareto(b=5).rvs(size=n, random_state=rng2)

    return U, U_tilde, N_dep, S_dep, N_ind, S_ind


U, U_tilde, N_dep, S_dep, N_ind, S_ind = simulate_insurance(theta_clayton, n_ins)

# Scatter: Clayton vs Reverse Clayton
fig4 = make_subplots(rows=1, cols=2,
                     subplot_titles=("Clayton (lower tail dep.)", "Reverse Clayton (upper tail dep.)"))
fig4.add_trace(go.Scatter(
    x=U_tilde[0], y=U_tilde[1], mode="markers",
    marker=dict(color="steelblue", size=2, opacity=0.3), showlegend=False
), row=1, col=1)
fig4.add_trace(go.Scatter(
    x=U[0], y=U[1], mode="markers",
    marker=dict(color="darkorange", size=2, opacity=0.3), showlegend=False
), row=1, col=2)
fig4.update_xaxes(title_text="U₁", range=[0, 1])
fig4.update_yaxes(title_text="U₂", range=[0, 1])
st.plotly_chart(fig4, use_container_width=True)

# Insurance VaR comparison
X_dep = N_dep * S_dep
X_ind = N_ind * S_ind

alpha_ins = st.select_slider("VaR confidence level:", options=[0.95, 0.975, 0.99, 0.995], value=0.995)
var_dep = np.quantile(X_dep, alpha_ins)
var_ind = np.quantile(X_ind, alpha_ins)

c1, c2, c3 = st.columns(3)
c1.metric(f"VaR (dependent, {alpha_ins})", f"{var_dep/1000:.3f} Mio €")
c2.metric(f"VaR (independent, {alpha_ins})", f"{var_ind/1000:.3f} Mio €")
c3.metric("Difference (tail dep. premium)", f"{(var_dep - var_ind)/1000:.3f} Mio €",
          delta=f"+{(var_dep/var_ind - 1)*100:.1f}% vs independent")

fig5 = make_subplots(rows=1, cols=2,
                     subplot_titles=("Dependent claims (X = N·S)", "Independent claims (X_ind)"))
max_x = min(np.percentile(np.concatenate([X_dep, X_ind]), 99.5), max(X_dep.max(), X_ind.max()))
for col_idx, (X, color) in enumerate([(X_dep, "steelblue"), (X_ind, "crimson")], 1):
    counts, bins = np.histogram(X[X <= max_x], bins=60)
    fig5.add_trace(go.Bar(
        x=(bins[:-1] + bins[1:]) / 2, y=counts,
        marker_color=color, opacity=0.7, showlegend=False
    ), row=1, col=col_idx)
fig5.update_xaxes(title_text="Total Claim X (€)")
fig5.update_yaxes(title_text="Frequency")
st.plotly_chart(fig5, use_container_width=True)

st.markdown(f"""
**Key result:** At the {alpha_ins} confidence level, the upper tail dependence increases the
required capital reserve by **{(var_dep/var_ind - 1)*100:.1f}%** vs the independence assumption
({var_dep/1000:.2f} vs {var_ind/1000:.2f} Mio €). Ignoring tail dependence leads to systematic
**underestimation of catastrophic risk** — a crucial insight for insurance solvency (Solvency II).
""")
st.write("---")


# ============================================================
# SECTION 5: Kendall's τ and Gaussian Copula Fitting
# ============================================================
st.header("5. Fitting a Gaussian Copula to DAX Returns")
st.markdown(r"""
To fit a Gaussian copula to real data, we need to estimate the correlation matrix $R$.
**Kendall's $\tau$** is a robust rank-based correlation measure:

$$\hat{\varrho}_\tau(X_i, X_j) = \frac{2}{n(n-1)} \sum_{k < l} \text{sign}(X_{k,i} - X_{l,i}) \cdot \text{sign}(X_{k,j} - X_{l,j})$$

For a Gaussian copula, the link between Kendall's $\tau$ and the linear correlation is:

$$\varrho_L = \sin\!\left(\frac{\pi}{2} \hat{\varrho}_\tau\right)$$

**Step 1:** Transform log-returns to uniform marginals via the empirical CDF:
$\hat{F}_i(X_{n,i}) = \frac{1}{N}\sum_{k=1}^N \mathbf{1}_{X_{k,i} \leq X_{n,i}}$
""")


@st.cache_data
def fit_gaussian_copula(lr):
    n = lr.shape[0]
    d = lr.shape[1]

    # Transformed returns (pseudo-observations)
    transformed = np.zeros_like(lr)
    for i in range(d):
        transformed[:, i] = np.mean(lr[:, i, None] >= lr[:, i], axis=1)

    # Kendall's tau matrix
    tau = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            if i == j:
                tau[i, j] = 1.0
            elif i < j:
                x, y = lr[:, i], lr[:, j]
                idx = np.arange(n)
                signs = np.sign(
                    np.subtract.outer(x, x) * np.subtract.outer(y, y)
                )
                tau[i, j] = signs[np.triu_indices(n, k=1)].mean()
                tau[j, i] = tau[i, j]

    # Gaussian copula R matrix
    R = np.sin(np.pi / 2 * tau)
    np.fill_diagonal(R, 1.0)

    return transformed, tau, R


transformed, kendall_tau_mat, R_gauss = fit_gaussian_copula(lr)

# Show scatter of transformed returns (BMW vs SAP)
pair_options = [(i, j) for i in range(5) for j in range(i+1, 5)]
pair_labels = [f"{STOCK_NAMES[i]} vs {STOCK_NAMES[j]}" for i, j in pair_options]
pair_choice = st.selectbox("Show pseudo-observations for pair:", pair_labels, index=0)
i_sel, j_sel = pair_options[pair_labels.index(pair_choice)]

fig6 = go.Figure()
fig6.add_trace(go.Scatter(
    x=transformed[:, i_sel], y=transformed[:, j_sel],
    mode="markers", name=pair_choice,
    marker=dict(color="steelblue", size=3, opacity=0.4)
))
fig6.update_layout(
    xaxis_title=f"F̂({STOCK_NAMES[i_sel]})", yaxis_title=f"F̂({STOCK_NAMES[j_sel]})",
    xaxis=dict(range=[0, 1], showgrid=True), yaxis=dict(range=[0, 1], showgrid=True)
)
st.plotly_chart(fig6, use_container_width=True)

# Heatmaps side by side
st.subheader("Estimated Kendall's τ and Gaussian Copula Correlation Matrix R")

fig7 = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Kendall's τ matrix", "Gaussian Copula R = sin(π/2 · τ)")
)
for col_idx, (mat, cmin, cmax) in enumerate([(kendall_tau_mat, -1, 1), (R_gauss, -1, 1)], 1):
    fig7.add_trace(go.Heatmap(
        z=mat, x=STOCK_NAMES, y=STOCK_NAMES,
        colorscale="RdBu", zmin=cmin, zmax=cmax,
        text=np.round(mat, 3), texttemplate="%{text}",
        showscale=(col_idx == 2)
    ), row=1, col=col_idx)
fig7.update_layout(height=380)
st.plotly_chart(fig7, use_container_width=True)

st.markdown(f"""
**Findings:** All five DAX stocks are **positively correlated** — the portfolio shares broad
market exposure. The strongest dependence is between BMW and Volkswagen
($\\hat{{\\varrho}}_\\tau \\approx {kendall_tau_mat[0,2]:.2f}$), consistent with both being
automotive manufacturers. SAP (tech sector) shows weaker but still positive correlation with
the industrials, reflecting common macro-factor exposure rather than sector-specific co-movement.

The small deviation of diagonal entries from 1 is an artifact of duplicate zero log-returns
(weekends/holidays) that create a discrete mass point — a known caveat noted in the lecture.
""")
