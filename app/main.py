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
#btt-btn {
    position: fixed;
    bottom: 2.5rem;
    right: 2rem;
    z-index: 999999;
    background-color: #262730;
    color: #fafafa;
    border: 1px solid #888;
    border-radius: 50%;
    width: 2.8rem;
    height: 2.8rem;
    font-size: 1.4rem;
    line-height: 2.8rem;
    text-align: center;
    cursor: pointer;
    opacity: 0.85;
    transition: opacity 0.2s, border-color 0.2s;
}
#btt-btn:hover { opacity: 1; border-color: #fff; }
</style>

<button id="btt-btn" title="Back to top">↑</button>

<script>
(function() {
    function scrollTop() {
        // Try every plausible Streamlit scroll container
        var selectors = [
            '[data-testid="stAppViewContainer"]',
            '[data-testid="stApp"]',
            '.main',
            '.block-container'
        ];
        selectors.forEach(function(sel) {
            var el = document.querySelector(sel);
            if (el) el.scrollTop = 0;
        });
        document.documentElement.scrollTop = 0;
        document.body.scrollTop = 0;
        window.scrollTo(0, 0);
    }

    function wire() {
        var btn = document.getElementById('btt-btn');
        if (btn && !btn._bttWired) {
            btn.addEventListener('click', scrollTop);
            btn._bttWired = true;
        }
    }

    wire();
    setTimeout(wire, 300);
    setTimeout(wire, 1000);
})();
</script>
""", unsafe_allow_html=True)
