import streamlit as st

st.set_page_config(page_title="RiskLearn – Home", layout="wide", page_icon="🛡️")

with st.sidebar:
    st.markdown("# 🛡️ RiskLearn")
    st.caption("Risk Management Concepts in Python")
    st.divider()

col1, col2 = st.columns([2, 1])

with col1:
    st.title("RiskLearn Dashboard")
    st.markdown("""
    An **interactive learning dashboard** for risk management concepts in Python.
    Content follows the *'Risk Management'* curriculum at **CAU Kiel**.
    """)

    st.info("""
    **Academic Resource** — Course material by **Prof. Dr. Jan Kallsen**:
    [University of Kiel Lecture Notes](https://www.math.uni-kiel.de/finmath/de/personen/kallsen/lec_notes)
    """, icon="📖")

with col2:
    st.markdown("### Quick Navigation")
    with st.container(border=True):
        if st.button("📊 Explore DAX Data", use_container_width=True):
            st.switch_page("pages/1_Explore_Data.py")
        if st.button("📉 Loss Operator", use_container_width=True):
            st.switch_page("pages/2_1_Losses.py")
        if st.button("📏 Risk Measures (VaR / ES)", use_container_width=True):
            st.switch_page("pages/2_2_Risk_Measures.py")
        if st.button("🔍 Backtesting", use_container_width=True):
            st.switch_page("pages/3_Backtesting.py")
        if st.button("📈 Extreme Value Theory", use_container_width=True):
            st.switch_page("pages/4_Extreme_Value_Theory.py")
    st.caption("Or use the sidebar to navigate between modules.")

st.divider()

st.subheader("📚 Curriculum Coverage")

with st.expander("Chapter 1 — Foundations & Loss Distributions", expanded=False):
    st.markdown("""
    - **Exploratory Data Analysis:** DAX log returns, fat tails, Monte Carlo simulation
    - **Loss Operator:** Risk factors, nonlinear vs. linearised portfolio losses
    """)

with st.expander("Chapter 2 — Risk Measures & Backtesting", expanded=False):
    st.markdown("""
    - **Standard Deviation:** Classical risk measure, limitations for tail risk
    - **Value at Risk (VaR):** Parametric (normal) and historical simulation, rolling window
    - **Expected Shortfall (ES):** Coherent risk measure, Basel IV, VaR vs ES comparison
    - **Backtesting:** Visual exceedance analysis, binomial test, multi-level comparison
    """)

with st.expander("Chapter 3 — Extreme Value Theory", expanded=False):
    st.markdown("""
    - **Heavy Tails:** QQ plots vs. Normal / Student-t distributions
    - **Hill Estimator:** Tail index estimation and Hill plot
    - **Mean Excess Plot:** Threshold selection for POT method
    - **POT / GPD:** Generalized Pareto fitting, EVT-based VaR & ES, tail CDF comparison
    """)

st.divider()

st.caption("""
**Disclaimer:** This dashboard is an academic project for educational purposes following the
'Risk Management' curriculum at **CAU Kiel**. All models and calculations are not intended
for financial advice or live trading decisions.
""")
