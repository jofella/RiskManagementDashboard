import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy import stats

from util.data_utils import load_dax_index, get_log_returns

WINDOW = 252

st.title("🔍 Backtesting VaR")
st.markdown("""
A risk model that cannot be empirically validated is of limited regulatory and practical value.
**Backtesting** is the process of systematically comparing a model's predictions against
realised outcomes — asking: *does our VaR model actually contain losses at the stated frequency?*

### The Fundamental Idea
If $\text{VaR}_\alpha$ is correctly specified, then by definition:

$$P(L_{n+1} > \text{VaR}_\alpha^{(n)}) = 1 - \alpha$$

This means that on each day, the realised loss exceeds the predicted VaR with probability
$1 - \alpha$, independently of all other days. In a sample of $n$ observations, the number of
**exceedances** (also called *VaR violations*) should therefore follow:

$$N \sim \text{Bin}(n,\, 1-\alpha)$$

Backtesting tests whether the observed exceedance count is statistically consistent with this
theoretical distribution — providing a formal model validation procedure.

### Regulatory Context: The Basel Traffic Light
Under **Basel II/III**, banks must backtest their internal VaR models daily against
a 250-day rolling window at the 99% confidence level. The number of violations
determines the **multiplicative factor** applied to capital requirements:

| Zone | Violations (250 days) | Capital multiplier |
|---|---|---|
| 🟢 Green | 0 – 4 | 3.0 (baseline) |
| 🟡 Yellow | 5 – 9 | 3.4 – 3.85 |
| 🔴 Red | ≥ 10 | 4.0 (maximum) |

Too few violations can also be problematic — it may indicate an **over-conservative** model
that ties up unnecessary capital.

### Two Approaches Used Here
1. **Visual backtesting** — plot rolling VaR against realised losses; identify exceedances and
   check for temporal clustering (clustered violations suggest the model fails to capture
   volatility dynamics).
2. **Statistical backtesting (Kupiec test)** — formally test $H_0: p = 1 - \alpha$ using the
   binomial distribution; compute p-value and render a pass/fail verdict.
""")
st.write("---")


# --- Data ---
data = load_dax_index()
losses = -np.diff(data)
T = len(losses)
t_axis = np.arange(T)


# --- Settings ---
col1, col2, col3 = st.columns(3)
with col1:
    alpha = st.select_slider("Confidence level α:", options=[0.90, 0.95, 0.975, 0.99], value=0.99)
with col2:
    method = st.radio("VaR method:", ["Normal (parametric)", "Historical Simulation"], horizontal=False)
with col3:
    window = st.slider("Rolling window (days):", min_value=100, max_value=500, value=252, step=10)

st.write("---")


# --- Compute rolling VaR ---
@st.cache_data
def compute_rolling_var(losses, alpha, window, method):
    var = np.full(len(losses), np.nan)
    for t in range(window, len(losses)):
        w = losses[t - window:t]
        if method == "Normal (parametric)":
            var[t] = np.mean(w) + np.std(w, ddof=1) * stats.norm.ppf(alpha)
        else:
            var[t] = np.percentile(w, alpha * 100)
    return var


var_rolling = compute_rolling_var(losses, alpha, window, method)

valid_mask = ~np.isnan(var_rolling)
n_valid = np.sum(valid_mask)
n_exceed = np.sum((losses > var_rolling) & valid_mask)
exceed_rate = n_exceed / n_valid
expected_rate = 1 - alpha
expected_count = n_valid * expected_rate

exceedance_idx = np.where((losses > var_rolling) & valid_mask)[0]


# === SECTION 1: Visual Backtesting ===
st.header("1. Visual Backtesting")
st.caption(
    "Orange line = rolling VaR estimate. Red dots = exceedances (days where actual loss > VaR). "
    "A well-calibrated model shows exceedances scattered uniformly, not clustered."
)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=t_axis, y=losses,
    mode="lines", name="Daily Loss",
    line=dict(color="steelblue", width=1), opacity=0.6
))
fig.add_trace(go.Scatter(
    x=t_axis, y=var_rolling,
    mode="lines", name=f"VaR_{alpha} ({method[:6]})",
    line=dict(color="orange", width=2)
))
fig.add_trace(go.Scatter(
    x=exceedance_idx, y=losses[exceedance_idx],
    mode="markers", name="Exceedance",
    marker=dict(color="red", size=6, symbol="circle")
))
fig.update_layout(
    xaxis_title="Trading Day", yaxis_title="Loss (index points)",
    legend=dict(x=0, y=1), xaxis=dict(showgrid=True), yaxis=dict(showgrid=True)
)
st.plotly_chart(fig, use_container_width=True)

# Exceedances over time (bar chart by year)
st.subheader("Exceedances per Year")
st.caption("Exceedances should be uniformly distributed. Clustering indicates model stress or regime change.")

years_per_day = 252
n_years = T // years_per_day
year_labels = [f"Y{i+1}" for i in range(n_years)]
exceed_by_year = [
    np.sum((losses[i * years_per_day:(i+1) * years_per_day] >
            var_rolling[i * years_per_day:(i+1) * years_per_day]) &
           valid_mask[i * years_per_day:(i+1) * years_per_day])
    for i in range(n_years)
]

fig_bar = go.Figure(go.Bar(
    x=year_labels, y=exceed_by_year,
    marker_color=["crimson" if v > expected_rate * years_per_day * 2 else "steelblue" for v in exceed_by_year],
    name="Exceedances"
))
fig_bar.add_hline(
    y=expected_rate * years_per_day, line_dash="dash", line_color="orange",
    annotation_text=f"Expected ({expected_rate*years_per_day:.1f}/yr)"
)
fig_bar.update_layout(xaxis_title="Year (approx.)", yaxis_title="# Exceedances")
st.plotly_chart(fig_bar, use_container_width=True)


# === SECTION 2: Statistical Backtesting ===
st.write("---")
st.header("2. Statistical Backtesting (Binomial Test)")
st.markdown(f"""
Under the null hypothesis $H_0$ that the VaR model is correctly specified,
each day is independently a VaR exceedance with probability $p_0 = 1-\\alpha = {expected_rate}$.

The number of exceedances $N$ follows a **Binomial distribution**:

$$N \\sim \\text{{Bin}}(n, p_0), \\quad n = {n_valid}, \\quad p_0 = {expected_rate}$$

We test $H_0: p = p_0$ against $H_1: p \\neq p_0$ (two-sided).
""")

# Binomial test
p_value = stats.binomtest(n_exceed, n_valid, expected_rate, alternative="two-sided").pvalue
z_stat = (n_exceed - expected_count) / np.sqrt(n_valid * expected_rate * alpha)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Observed exceedances", f"{n_exceed}")
col2.metric("Expected exceedances", f"{expected_count:.1f}")
col3.metric("Exceedance rate", f"{exceed_rate*100:.2f}%", delta=f"{(exceed_rate - expected_rate)*100:+.2f}pp")
col4.metric("p-value", f"{p_value:.4f}", delta="PASS ✓" if p_value > 0.05 else "FAIL ✗",
            delta_color="normal" if p_value > 0.05 else "inverse")

st.write("---")

# Binomial distribution plot
st.subheader("Binomial Distribution under H₀")
st.caption("The red dashed line shows the observed exceedance count. If it falls in the tails, the model is rejected.")

k_range = np.arange(0, int(expected_count * 5))
binom_pmf = stats.binom.pmf(k_range, n_valid, expected_rate)

# Critical region at 5% significance
lower_crit = stats.binom.ppf(0.025, n_valid, expected_rate)
upper_crit = stats.binom.ppf(0.975, n_valid, expected_rate)

fig3 = go.Figure()
colors = [
    "crimson" if (k <= lower_crit or k >= upper_crit) else "steelblue"
    for k in k_range
]
fig3.add_trace(go.Bar(
    x=k_range, y=binom_pmf,
    marker_color=colors, name="Bin(n, p₀) PMF",
    opacity=0.8
))
fig3.add_vline(
    x=n_exceed, line_dash="dash", line_color="red", line_width=2,
    annotation_text=f"Observed: {n_exceed}", annotation_position="top right"
)
fig3.add_vline(
    x=expected_count, line_dash="dot", line_color="orange", line_width=1.5,
    annotation_text=f"Expected: {expected_count:.0f}", annotation_position="top left"
)
fig3.update_layout(
    xaxis_title="Number of Exceedances", yaxis_title="Probability",
    xaxis=dict(range=[max(0, expected_count - 4 * np.sqrt(n_valid * expected_rate * alpha)),
                      expected_count + 4 * np.sqrt(n_valid * expected_rate * alpha)])
)
st.plotly_chart(fig3, use_container_width=True)

# Verdict
if p_value > 0.05:
    st.success(
        f"**Model not rejected** (p = {p_value:.4f} > 0.05). "
        f"The {method} VaR at α={alpha} is statistically consistent with the data.",
        icon="✅"
    )
else:
    st.error(
        f"**Model rejected** (p = {p_value:.4f} ≤ 0.05). "
        f"The {method} VaR at α={alpha} produces {'too many' if n_exceed > expected_count else 'too few'} exceedances "
        f"({n_exceed} observed vs {expected_count:.0f} expected).",
        icon="❌"
    )

st.write("---")


# === SECTION 3: Comparison across confidence levels ===
st.header("3. Comparison Across Confidence Levels")
st.caption("How well does the model perform at different confidence levels simultaneously?")

alphas = [0.90, 0.95, 0.975, 0.99]
rows = []
for a in alphas:
    v = compute_rolling_var(losses, a, window, method)
    vm = ~np.isnan(v)
    n_v = np.sum(vm)
    n_e = np.sum((losses > v) & vm)
    p_v = stats.binomtest(n_e, n_v, 1 - a, alternative="two-sided").pvalue
    rows.append({
        "α": a, "Expected rate": f"{(1-a)*100:.1f}%",
        "Observed exceedances": n_e, "Expected exceedances": f"{n_v*(1-a):.1f}",
        "Observed rate": f"{n_e/n_v*100:.2f}%",
        "p-value": f"{p_v:.4f}",
        "Result": "✅ Pass" if p_v > 0.05 else "❌ Fail"
    })

st.dataframe(pd.DataFrame(rows).set_index("α"), use_container_width=True)

st.markdown("""
**Interpretation guide:**
- **Pass** at all levels → model is well-calibrated
- **Fail at high α (0.99)** → model underestimates extreme tail risk (common with normal assumption)
- **Clustered exceedances** → violations are not i.i.d.; GARCH models or conditional VaR would help
""")
