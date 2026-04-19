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
        st.Page("pages/6_GARCH.py",               title="GARCH Model",            icon="📉"),
    ],
}

pg = st.navigation(pages)
pg.run()

st.markdown("""
<style>
#back-to-top {
    position: fixed;
    bottom: 2.5rem;
    right: 2rem;
    z-index: 9999;
    background-color: #262730;
    color: #fafafa;
    border: 1px solid #555;
    border-radius: 50%;
    width: 2.8rem;
    height: 2.8rem;
    font-size: 1.3rem;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    opacity: 0;
    transition: opacity 0.3s ease, border-color 0.2s ease;
    pointer-events: none;
}
#back-to-top.visible {
    opacity: 0.85;
    pointer-events: auto;
}
#back-to-top:hover {
    opacity: 1;
    border-color: #aaa;
}
</style>
<button id="back-to-top" title="Back to top" onclick="window.scrollTo({top:0,behavior:'smooth'})">↑</button>
<script>
window.addEventListener('scroll', function() {
    const btn = document.getElementById('back-to-top');
    if (btn) {
        if (window.scrollY > 300) {
            btn.classList.add('visible');
        } else {
            btn.classList.remove('visible');
        }
    }
});
</script>
""", unsafe_allow_html=True)
