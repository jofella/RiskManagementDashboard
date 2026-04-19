import streamlit as st
import streamlit.components.v1 as components

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

components.html("""
<script>
(function() {
    var doc = window.parent.document;

    // Inject button + style into parent page if not already there
    if (!doc.getElementById('btt-btn')) {
        var style = doc.createElement('style');
        style.textContent = [
            '#btt-btn {',
            '  position: fixed; bottom: 2.5rem; right: 2rem; z-index: 999999;',
            '  background: #262730; color: #fafafa; border: 1px solid #888;',
            '  border-radius: 50%; width: 2.8rem; height: 2.8rem;',
            '  font-size: 1.4rem; line-height: 2.8rem; text-align: center;',
            '  cursor: pointer; opacity: 0.85; transition: opacity 0.2s;',
            '}',
            '#btt-btn:hover { opacity: 1; border-color: #fff; }'
        ].join('');
        doc.head.appendChild(style);

        var btn = doc.createElement('button');
        btn.id = 'btt-btn';
        btn.title = 'Back to top';
        btn.textContent = '\u2191';
        btn.addEventListener('click', function() {
            // Reset scrollTop on every element that has scrolled
            doc.querySelectorAll('*').forEach(function(el) {
                try { if (el.scrollTop > 0) el.scrollTop = 0; } catch(e) {}
            });
            window.parent.scrollTo(0, 0);
        });
        doc.body.appendChild(btn);
    }
})();
</script>
""", height=0)
