# All commonly used function like load, etc.
import numpy as np
import pandas as pd
import streamlit as st


# 1. Get log-returns
def get_log_returns(data):
    return np.diff(np.log(data))


# 2. Load (single-stock) data
@st.cache_data
def load_single_stock_data(path):
    """Load single-stock data from CSV."""
    return np.genfromtxt(path, usecols=(1), delimiter=",", skip_header=1)



# --- 1. Data Loading ---
class DataLoader:
    def __init__(self, source: str):
        self.source = source
    
    def LoadCSV(self) -> pd.DataFrame:
        df = pd.read_csv(self.source)
        return df




class DataTrafo:
    def __init__(self):
        pass



class RiskModel:
    def __init__(self):
        pass




loader = DataLoader("data/DAX_index.csv")
raw_df = loader.load_csv()