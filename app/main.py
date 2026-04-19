import streamlit as st

st.set_page_config(page_title="RiskLearn", layout="wide", page_icon="🛡️")

# Anchor target for back-to-top link
st.markdown('<div id="top"></div>', unsafe_allow_html=True)

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
        st.Page("pages/6_GARCH.py",               title="GARCH Model",            icon="📉"),
    ],
}

pg = st.navigation(pages)
pg.run()

st.markdown("""
<style>
.btt-link {
    position: fixed;
    bottom: 2.5rem;
    right: 2rem;
    z-index: 999999;
    background: #262730;
    color: #fafafa !important;
    border: 1px solid #888;
    border-radius: 50%;
    width: 2.8rem;
    height: 2.8rem;
    font-size: 1.4rem;
    line-height: 2.8rem;
    text-align: center;
    text-decoration: none !important;
    opacity: 0.85;
    display: block;
}
.btt-link:hover { opacity: 1; border-color: #fff; }
</style>
<a class="btt-link" href="#top" title="Back to top">↑</a>
""", unsafe_allow_html=True)
