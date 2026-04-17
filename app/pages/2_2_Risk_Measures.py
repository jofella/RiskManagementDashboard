import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

from risk_measure import standard_deviation, VaR, ES

st.title("📏 Measuring Risk")
st.markdown("""
Having quantified risk through the loss operator, we now need to **summarise the entire loss
distribution in a single number** that can serve as a capital requirement. This number is
produced by a **risk measure** — a functional $\varrho$ mapping a random loss $L$ to a real value
representing the required buffer capital.

### What Makes a Good Risk Measure? Coherence
Artzner, Delbaen, Eber, and Heath (1999) proposed four axioms that any *rational* risk measure
should satisfy — giving rise to the concept of a **coherent risk measure**:

| Axiom | Meaning |
|---|---|
| **Monotonicity** | If $L_1 \leq L_2$ a.s., then $\varrho(L_1) \leq \varrho(L_2)$ |
| **Translation invariance** | Adding cash $m$ reduces risk: $\varrho(L - m) = \varrho(L) - m$ |
| **Positive homogeneity** | Scaling the portfolio scales risk: $\varrho(\lambda L) = \lambda\,\varrho(L)$ |
| **Subadditivity** | Diversification cannot increase risk: $\varrho(L_1 + L_2) \leq \varrho(L_1) + \varrho(L_2)$ |

**Subadditivity** is the most economically important axiom — it ensures that combining two
portfolios never requires *more* capital than holding them separately, consistent with the
intuition that diversification reduces risk.

### The Three Measures Covered Here
- **Standard Deviation** — simple, symmetric, but fails subadditivity for skewed distributions
  and is blind to tail shape.
- **Value at Risk (VaR)** — the regulatory standard since Basel II; intuitive but **not coherent**
  (it violates subadditivity) and ignores the severity of losses beyond the threshold.
- **Expected Shortfall (ES)** — coherent, captures tail severity, and has replaced VaR in
  Basel IV (FRTB) for internal model approaches.

Select a risk measure below to explore its definition, properties, and application to DAX data.
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
