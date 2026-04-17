# 📊 Risk Management Dashboard

This project was developed during my master’s studies as part of a **Risk Management** course.  
It serves both as a review of theoretical concepts and an exercise in building interactive dashboards with **Streamlit**.  
It is also my **first application-oriented project** using Streamlit and modular Python structuring.

🔗 **Live dashboard:**  
https://riskmanagementdashboard-ahcrwbuaieyzxu4hvzryih.streamlit.app

---

## 📖 Project Overview

The **Risk Management Dashboard** is an educational, interactive application designed to make abstract financial risk models more tangible through visualization and simulation.

Originally conceived during coursework, the project focuses on:

- Translating theoretical risk management concepts into a functional analytical tool  
- Presenting financial risk measures through interactive visualizations  
- Demonstrating clean, modular Python project structuring  

---

## 🎯 Key Objectives

- **Theoretical Application**  
  Implement core concepts from risk management lectures (e.g., Value at Risk)

- **Software Engineering**  
  Build a modular, maintainable codebase with clear separation between analytical logic and frontend presentation

---

## ⚙️ Features

### Chapter 1 — Foundations & Loss Distributions
- Log-return analysis of the DAX index (2000–2024): mean, std, skewness, excess kurtosis
- Empirical vs. normal distribution comparison with interactive histogram
- Monte Carlo price simulation (Geometric Brownian Motion)
- Formal loss operator: nonlinear vs. linearised (Taylor) portfolio losses for a 5-stock DAX portfolio

### Chapter 2 — Risk Measures
- **Standard Deviation** as a risk measure with interactive risk-appetite parameter $c$
- **Value at Risk (VaR)**: parametric (normal) and historical simulation, rolling 252-day window, exceedance markers
- **Expected Shortfall (ES)**: rolling VaR vs ES overlay, Basel IV coherence motivation, distribution illustration
- Backtesting: visual exceedance plot, exceedances-per-year bar chart, Kupiec binomial test with p-value and Basel traffic light logic, multi-confidence-level comparison table

### Chapter 3 — Extreme Value Theory
- QQ plots against Normal and Student-t distributions to diagnose fat tails
- Hill estimator and Hill plot for tail index $\xi$ estimation
- Mean Excess Plot (MEP) for threshold selection
- Peak-over-Threshold (POT): GPD fitting via MLE, EVT-based VaR & ES table vs normal benchmark, log-scale tail CDF comparison

### Chapter 4 — Copulas & Dependence Structures
- Rolling Pearson vs. Spearman rank correlation (BMW vs. Volkswagen, 252-day window)
- Gaussian copula: Cholesky sampling with interactive $\rho$ and sample size
- Normal mixture copula: interactive mixing weight and component correlations
- Clayton / Reverse Clayton copula via Marshall-Olkin sampling; insurance application with VaR comparison (dependent vs. independent, configurable $\vartheta$ and confidence level)
- Kendall's $\tau$ estimation → Gaussian copula correlation matrix $R$ for all 5 DAX stocks; pseudo-observation scatter and dual heatmap

---

## 🗂️ Project Structure

```
RiskLearnDashboard/
│
├── app/
│   ├── main.py                        # Navigation entrypoint (st.navigation)
│   └── pages/
│       ├── home.py                    # Landing page
│       ├── 1_Explore_Data.py          # Chapter 1: Log returns & simulation
│       ├── 2_1_Losses.py              # Chapter 1: Loss operator
│       ├── 2_2_Risk_Measures.py       # Chapter 2: SD / VaR / ES
│       ├── 3_Backtesting.py           # Chapter 2: Backtesting
│       ├── 4_Extreme_Value_Theory.py  # Chapter 3: EVT / POT / GPD
│       ├── 5_Copulas.py               # Chapter 4: Copulas
│       └── risk_measure/
│           ├── standard_deviation.py
│           ├── VaR.py
│           └── ES.py
│
├── util/
│   └── data_utils.py                  # Cached data loading & log-return helper
│
├── data/
│   ├── DAX_index.csv                  # DAX daily closing prices 2000–2024
│   └── DAX_companies.csv              # BMW, SAP, VW, Continental, Siemens
│
├── Sources/                           # Course notebooks (Weeks 1–7)
└── requirements.txt
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/jofella/RiskLearnDashboard.git
cd RiskLearnDashboard
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app/main.py
```
