import streamlit as st

st.set_page_config(page_title="RiskLearn", layout="wide", page_icon="🛡️")

# Sidebar — runs on every page via st.navigation
with st.sidebar:
    st.markdown("## 🛡️ RiskLearn")
    st.caption("Risk Management · CAU Kiel")
    st.divider()

pages = {
    "": [
        st.Page("pages/home.py", title="Home", icon="🏠", default=True),
    ],
    "Chapters": [
        st.Page("pages/1_Explore_Data.py",        title="Explore Data",          icon="📊"),
        st.Page("pages/2_1_Losses.py",            title="Loss Operator",          icon="📉"),
        st.Page("pages/2_2_Risk_Measures.py",     title="Risk Measures",          icon="📏"),
        st.Page("pages/3_Backtesting.py",         title="Backtesting",            icon="🔍"),
        st.Page("pages/4_Extreme_Value_Theory.py",title="Extreme Value Theory",   icon="📈"),
        st.Page("pages/5_Copulas.py",             title="Copulas",                icon="🔗"),
    ],
}

pg = st.navigation(pages)
pg.run()
