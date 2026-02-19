import streamlit as st

st.set_page_config(page_title="RiskLearn – Home", layout="wide", page_icon="🛡️")

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("# 🛡️ RiskLearn")
    st.caption("Risk Management Concepts in Python")
    st.divider()

# --- MAIN CONTENT ---
# Using columns to center the content slightly or add a visual element
col1, col2 = st.columns([2, 1])

with col1:
    st.title("RiskLearn Dashboard")
    st.markdown("""
    This is a small **vibe coding project** for learning risk management concepts in Python. 
    The content is based on the course materials from the *'Risk Management'* class at **CAU Kiel**.
    """)
    
    # Using a stylized info box for the academic reference
    st.info(f"""
    **Academic Resource** Class material by **Prof. Dr. Jan Kallsen** can be found here:  
    [University of Kiel Lecture Notes](https://www.math.uni-kiel.de/finmath/de/personen/kallsen/lec_notes)
    """, icon="📖")

with col2:
    st.markdown("### Quick Actions")
    
    # Using a container to give it a subtle border
    with st.container(border=False):
        if st.button("📊 Explore DAX Data", use_container_width=True):
            st.switch_page("pages/1_Explore_Data.py")
            
        if st.button("🛡️ Risk Measures", use_container_width=True):
            st.switch_page("pages/2_Risk_Measures.py")
            
        if st.button("📉 EVT Analysis", use_container_width=True):
            st.switch_page("pages/Extreme_Value_Theory.py")
            
    st.caption("Jump directly to specific course modules.")


st.divider()


st.subheader("📚 Curriculum Coverage")

with st.expander("Section 1: Foundations & Loss Distributions"):
    st.write("""
    - **Exploratory Data Analysis:** Handling financial time series (DAX index).
    - **Loss Distributions:** Modeling empirical vs. theoretical loss functions.
    """)

with st.expander("Section 2: Risk Metrics"):
    st.write("""
    - **Value-at-Risk (VaR):** Calculating the maximum potential loss over a time horizon.
    - **Expected Shortfall (ES):** Looking at the 'average' of the tail losses.
    """)

with st.expander("Section 3: Advanced Modeling"):
    st.write("""
    - **Extreme Value Theory (EVT):** Modeling the "Black Swan" events using Peak-over-Threshold.
    - **Multivariate Methods:** Analyzing dependencies between different assets (Copulas).
    """)
    

# --- FOOTER SECTION ---
st.divider()

st.caption("""
    **Disclaimer:** This dashboard is an academic project developed for educational purposes 
    following the 'Risk Management' curriculum at **CAU Kiel**. The calculations, 
    models (VaR, ES, EVT), and data visualizations provided are not intended for 
    financial advice or live trading decisions.
    """)