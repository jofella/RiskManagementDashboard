import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

from risk_measure import standard_deviation, VaR, ES

st.title("📏 Measuring Risk")
st.markdown("""
We have established a way to **quantify** risk through losses.
The next step is to **measure** risk and determine the required **buffer capital**.

Select a risk measure below to explore its definition, properties, and application to the DAX portfolio.
""")
st.write("---")

method = st.radio(
    "Choose a risk measure:",
    ["Standard Deviation", "Value at Risk (VaR)", "Expected Shortfall (ES)"],
    horizontal=True
)
st.write("---")

if method == "Standard Deviation":
    standard_deviation.render()
elif method == "Value at Risk (VaR)":
    VaR.render()
elif method == "Expected Shortfall (ES)":
    ES.render()
