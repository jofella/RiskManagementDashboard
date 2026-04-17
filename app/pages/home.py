import streamlit as st

st.title("RiskLearn Dashboard")
st.markdown("""
An **interactive learning dashboard** for risk management concepts in Python.
Content follows the *'Risk Management'* curriculum at **CAU Kiel**.
""")
st.info("""
**Academic Resource** — Course material by **Prof. Dr. Jan Kallsen**:
[University of Kiel Lecture Notes](https://www.math.uni-kiel.de/finmath/de/personen/kallsen/lec_notes)
""", icon="📖")

st.divider()
st.subheader("📚 Curriculum Coverage")

with st.expander("Chapter 1 — Foundations & Loss Distributions"):
    st.markdown("""
    - **Explore Data:** DAX log returns, fat tails, Monte Carlo simulation
    - **Loss Operator:** Risk factors, nonlinear vs. linearised portfolio losses
    """)

with st.expander("Chapter 2 — Risk Measures & Backtesting"):
    st.markdown("""
    - **Standard Deviation:** Classical risk measure, limitations for tail risk
    - **Value at Risk (VaR):** Parametric and historical simulation, rolling window
    - **Expected Shortfall (ES):** Coherent risk measure, Basel IV motivation
    - **Backtesting:** Visual exceedance analysis, binomial test, multi-level comparison
    """)

with st.expander("Chapter 3 — Extreme Value Theory"):
    st.markdown("""
    - **Heavy Tails:** QQ plots vs Normal / Student-t
    - **Hill Estimator:** Tail index and Hill plot
    - **Mean Excess Plot:** Threshold selection for the POT method
    - **POT / GPD:** Generalized Pareto fitting, EVT-based VaR & ES, tail CDF comparison
    """)

st.divider()
st.caption("""
**Disclaimer:** Academic project for educational purposes. Not intended for financial advice or live trading.
""")
