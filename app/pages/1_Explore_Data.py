import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from util.data_utils import load_dax_index, get_log_returns

# --- HEADER ---
st.title("📊 Explore Data")
st.markdown(r"""
Before building any risk model, we must understand the **statistical properties of financial returns**.
This section analyses daily log returns of the DAX index and confronts them with the most common
modelling assumption: that returns are independently and identically normally distributed (i.i.d. normal).

### The Log-Return Model
For a price process $(S_n)_{n \geq 0}$, the **log return** over period $n$ is defined as:

$$X_n = \log S_n - \log S_{n-1} = \log\left(\frac{S_n}{S_{n-1}}\right)$$

Log returns are preferred over simple returns $S_n/S_{n-1} - 1$ for several reasons:
they are **additively aggregable** over time ($X_1 + \cdots + X_n = \log(S_n/S_0)$), approximately
symmetric around zero, and naturally connected to the continuous-time model of Geometric Brownian Motion.

### The Normal Assumption — and Why It Fails
Under the classical **Black-Scholes** framework, log returns are assumed i.i.d. normal:
$X_n \sim \mathcal{N}(\mu, \sigma^2)$. This implies that the price process follows a
**Geometric Brownian Motion** and leads to analytically tractable pricing formulas.

However, empirical financial data consistently violates this assumption through a set of
well-documented **stylised facts**:
- **Leptokurtosis (fat tails):** Extreme returns occur far more frequently than the normal
  distribution predicts. Excess kurtosis $> 0$ is the statistical signature.
- **Volatility clustering:** Large price moves tend to be followed by large moves, small by small
  — a phenomenon captured by GARCH-type models but absent under i.i.d. normality.
- **Slight negative skewness:** Losses tend to be larger in magnitude than equivalent gains.

These features are not academic curiosities — they directly determine how often risk models
*underestimate* extreme losses, with profound consequences for capital requirements and solvency.
""")

with st.expander("📖 Intuition: Why log returns and not simple returns?"):
    st.markdown(r"""
    Suppose a stock is worth €100 today and €110 tomorrow. The **simple return** is
    $r = (110 - 100)/100 = 10\%$. The **log return** is $X = \ln(110/100) \approx 9.53\%$.

    For small moves they are nearly identical — but log returns have three structural advantages:

    **1. Time additivity.** If you hold for two days with log returns $X_1$ and $X_2$, the total
    log return is simply $X_1 + X_2 = \ln(S_2/S_0)$. With simple returns you'd have to multiply:
    $(1+r_1)(1+r_2) - 1$ — messy for longer horizons.

    **2. Symmetry.** A 50% loss followed by a 100% gain leaves you where you started.
    Simple returns are asymmetric around zero; log returns are not.

    **3. Continuous-time connection.** Under Geometric Brownian Motion (Black-Scholes),
    log returns are exactly normally distributed over any fixed interval. This makes them the
    natural input to all analytical risk models.

    **The practical cost:** log returns are not directly interpretable in euro terms — you need to
    exponentiate to get back to price changes.
    """)

with st.expander("📖 Intuition: What are stylised facts?"):
    st.markdown(r"""
    Stylised facts are **statistical regularities** that appear consistently across different
    assets, time periods, and markets — robust enough to be treated as empirical laws.

    | Stylised fact | What it means | Why it matters |
    |---|---|---|
    | **Fat tails (leptokurtosis)** | Extreme moves happen 10–100× more often than a normal distribution predicts | VaR models based on normality massively understate tail risk |
    | **Volatility clustering** | Turbulent days cluster together; calm periods cluster together | Risk is not constant — GARCH models are needed |
    | **Negative skewness** | Large negative moves are more common than large positive ones | Losses are asymmetric — put options are systematically expensive |
    | **Absence of autocorrelation in returns** | Yesterday's return tells you almost nothing about today's | Prices are (approximately) unpredictable — Efficient Market Hypothesis |
    | **Autocorrelation in squared returns** | Yesterday's *squared* return predicts today's | Volatility is predictable even when returns are not |

    The last two together explain why GARCH works: it models the *variance* process, not the
    returns themselves.
    """)

st.write("---")

# 1. Load Data
st.header("1. Load Data")
st.write("We use DAX daily closing prices (2000–2024) to compute log returns.")

data = load_dax_index()
lr = get_log_returns(data)

mu = np.mean(lr)
sigma = np.std(lr)

st.success(f"Loaded **{len(data):,}** daily price observations → **{len(lr):,}** log returns.")
st.write("---")

# 2. Summary Statistics
st.header("2. Summary Statistics")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Mean (μ)", f"{mu:.6f}")
col2.metric("Std Dev (σ)", f"{sigma:.6f}")
col3.metric("Skewness", f"{stats.skew(lr):.4f}")
col4.metric("Excess Kurtosis", f"{stats.kurtosis(lr):.4f}")

st.markdown(f"""
The sample mean of {mu:.6f} is statistically indistinguishable from zero over this horizon —
consistent with the Efficient Market Hypothesis that log returns are (approximately) unpredictable.
The annualised volatility is $\\hat{{\\sigma}} \\times \\sqrt{{252}} \\approx {np.std(lr)*np.sqrt(252)*100:.1f}\\%$,
within the typical range for developed equity indices (15–25% p.a. in calm periods, >40% in crises).

The excess kurtosis of **{stats.kurtosis(lr):.2f}** (vs. 0 for a normal distribution) is the
key diagnostic. For a Student-t distribution, excess kurtosis equals $6/(\\nu-4)$ for $\\nu > 4$,
implying an effective degrees-of-freedom of roughly $\\hat{{\\nu}} \\approx 4 + 6/{stats.kurtosis(lr):.2f} \\approx {4 + 6/stats.kurtosis(lr):.1f}$.
This is consistent with the heavy-tail estimates we will obtain later from the Hill estimator and GPD fitting.

The **negative skewness of {stats.skew(lr):.3f}** confirms the well-documented *leverage effect*:
large negative returns (crashes) are more frequent and more extreme than large positive returns
of the same magnitude. A normal distribution has zero skewness by definition, so it cannot
capture this asymmetry. For VaR and ES estimation, this means normal-based methods will
*underestimate* downside risk relative to what historical simulation or EVT-based methods produce.
""")
st.write("---")

# 3. Log Returns Over Time
st.header("3. Log Returns Over Time")

sim_returns = np.random.normal(mu, sigma, len(lr))
S_t = data[0] * np.exp(np.cumsum(sim_returns))

lr_df = pd.DataFrame({"Index": range(len(lr)), "Log Returns": lr})
sim_df = pd.DataFrame({"Index": range(len(sim_returns)), "Simulated Log Returns": sim_returns})
price_df = pd.DataFrame({"Index": range(len(data)), "Real DAX": data})
sim_price_df = pd.DataFrame({"Index": range(len(S_t)), "Simulated DAX": S_t})

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=lr_df["Index"], y=lr_df["Log Returns"],
    mode="lines", name="Empirical", line=dict(color="steelblue")
))
fig.add_trace(go.Scatter(
    x=sim_df["Index"], y=sim_df["Simulated Log Returns"],
    mode="lines", name="Simulated (Normal)", line=dict(color="crimson", dash="dash")
))
fig.update_layout(
    xaxis_title="Trading Day", yaxis_title="Log Return",
    legend=dict(x=0, y=1), xaxis=dict(showgrid=True), yaxis=dict(showgrid=True)
)
st.plotly_chart(fig, use_container_width=True)

st.markdown(r"""
**Findings:** Two features are immediately apparent when comparing the two series:

1. **Volatility clustering in the empirical series:** The DAX returns exhibit clearly identifiable
   calm and turbulent regimes. Major spikes correspond to the dot-com crash (2000–2002), the
   global financial crisis (2008–2009), the European sovereign debt crisis (2011), and the
   COVID-19 shock (March 2020). The autocorrelation of squared returns $\text{Corr}(X_t^2, X_{t-k}^2)$
   remains significantly positive for lags up to 100+ days — formal evidence against the i.i.d. assumption.

2. **Homoscedasticity in the simulated series:** The i.i.d. normal simulation shows no temporal
   structure — variance is constant throughout. Under this model, a 5% daily loss is equally
   likely on day 1 and day 5,000. This is the core failure of the Black-Scholes framework as
   a risk model. A constant-volatility VaR will be *too tight* during crises (when actual volatility
   is 2–4× its long-run average) and *too loose* during calm periods — exactly the wrong
   calibration at the exact wrong times.
""")

with st.expander("📖 Intuition: What is volatility clustering?"):
    st.markdown(r"""
    Look at the empirical return series above. You will notice that large moves (positive or
    negative) tend to arrive in *bursts*, separated by quieter periods. This is volatility
    clustering — formally captured by significant autocorrelation in $|X_t|$ and $X_t^2$
    even when $X_t$ itself shows no autocorrelation.

    **Why does it happen?** The main drivers are:
    - **Information arrival:** Major news (earnings, macro data, geopolitical shocks) triggers a
      cascade of re-pricing that plays out over several days.
    - **Leverage effect:** A falling stock price increases the debt-to-equity ratio, raising future
      volatility — creating a feedback loop.
    - **Market microstructure:** Liquidity dries up during turbulence, amplifying price moves.

    **Why does it matter for risk?** If today is a volatile day, tomorrow is also likely volatile.
    A constant-volatility VaR model will be *too loose* during crises (underestimates risk) and
    *too tight* during calm periods (over-allocates capital). GARCH(1,1) fixes this by making
    tomorrow's variance a function of today's shock.
    """)

st.write("---")

# 4. Distribution of Log Returns
st.header("4. Distribution of Log Returns")

x_range = np.linspace(min(lr), max(lr), 300)
norm_pdf = stats.norm.pdf(x_range, loc=mu, scale=sigma)

num_bins = st.slider("Number of bins:", 20, 500, 250, step=10)

fig = px.histogram(
    lr_df, x="Log Returns", nbins=num_bins,
    histnorm="probability density",
    color_discrete_sequence=["steelblue"],
    opacity=0.7
)
fig.add_trace(go.Scatter(
    x=x_range, y=norm_pdf,
    mode="lines", name="Normal PDF", line=dict(color="crimson", width=2)
))
fig.update_layout(xaxis_title="Log Return", yaxis_title="Density")
st.plotly_chart(fig, use_container_width=True)

st.markdown(r"""
**Findings:** The histogram reveals three simultaneous deviations from normality:

- **Taller peak (leptokurtosis):** The empirical density is more concentrated near zero than the
  normal curve — most days have very small moves.
- **Heavier tails:** The empirical distribution places substantially more probability mass beyond
  ±2σ. Under normality, a 4σ event has probability $\approx 6 \times 10^{-5}$ (once in 43 years
  of daily data). In the DAX sample, 4σ+ events occur dozens of times — several orders of
  magnitude more frequently than the normal predicts.
- **Slight left asymmetry:** The left tail (losses) is marginally heavier than the right tail
  (gains), consistent with the negative skewness reported in the summary statistics.

**Risk management consequence:** A normal-distribution VaR at 99% is calibrated to the
$(1-\alpha) = 1\%$ quantile of a Gaussian. If the true distribution has heavier tails,
the 1% quantile lies *further out* — meaning the normal model systematically underestimates
VaR. At 99.9% confidence (relevant for economic capital and Basel IV FRTB), this underestimation
can be by a factor of 2–3×.
""")

with st.expander("📖 Intuition: Excess kurtosis and fat tails"):
    st.markdown(r"""
    **Kurtosis** measures the weight in the tails of a distribution relative to its centre.
    The normal distribution has kurtosis = 3 (excess kurtosis = 0 by convention).

    - **Excess kurtosis > 0** (leptokurtic): more probability mass in the tails *and* the centre
      than the normal — the distribution is "peaked and fat-tailed."
    - **Excess kurtosis < 0** (platykurtic): thin tails — rare in finance.

    For the DAX returns, excess kurtosis is typically around 5–8. What does this mean concretely?

    Under a normal distribution, a 5σ event (five standard deviations from the mean) has
    probability ≈ $3 \times 10^{-7}$ — once every 13,000 years of daily trading.
    In practice, 5σ daily moves happen roughly **once every few years**. The normal distribution
    is not just slightly wrong in the tail — it is off by many orders of magnitude.

    **This is not a curiosity.** Institutions that use normal-distribution VaR at 99% are
    *systematically* underestimating the probability and severity of extreme losses.
    The 2008 financial crisis featured moves that were "25-sigma events" under normal assumptions —
    statistically impossible, but empirically observed.
    """)

st.write("---")

# 5. Monte Carlo Price Simulation
st.header("5. Monte Carlo Price Simulation")

st.info("This plot changes on every re-load since the simulated path is random.", icon="ℹ️")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=price_df["Index"], y=price_df["Real DAX"],
    mode="lines", name="Real DAX", line=dict(color="steelblue")
))
fig.add_trace(go.Scatter(
    x=sim_price_df["Index"], y=sim_price_df["Simulated DAX"],
    mode="lines", name="Simulated (GBM)", line=dict(color="crimson", dash="dash")
))
fig.update_layout(
    xaxis_title="Trading Day", yaxis_title="DAX Level",
    xaxis=dict(showgrid=True), yaxis=dict(showgrid=True)
)
st.plotly_chart(fig, use_container_width=True)

st.markdown(r"""
**Findings:** The GBM simulation and the real DAX index share the same unconditional mean
and variance (by construction — we calibrated $\mu$ and $\sigma$ to the data), yet they
look fundamentally different at the path level:

- The **simulated GBM path** evolves smoothly with no identifiable crisis periods — volatility is
  constant at $\hat{\sigma}\sqrt{252}$ throughout. Long-run drift is correctly captured.
- The **real DAX** shows sharp drops of 50%+ during 2002 and 2008, a rapid recovery, and a
  sudden crash in early 2020. These are not consistent with constant-volatility dynamics:
  the conditional distribution of returns changes dramatically over time.

**Why GBM still matters:** Despite this limitation, GBM remains the *benchmark* model for
option pricing (Black-Scholes), portfolio optimisation (mean-variance), and regulatory
capital under simplified frameworks. Understanding *where* it fails is as important as
knowing how to use it — and the failure mode is precisely the topic of the GARCH chapter.

**Heteroscedasticity test:** A formal test of constant variance (e.g. Engle's ARCH test)
would reject the null of no ARCH effects at essentially any standard significance level,
confirming that the GBM assumption is empirically untenable for the DAX.
""")
