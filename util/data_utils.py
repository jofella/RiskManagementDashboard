import os
import numpy as np
import streamlit as st

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


@st.cache_data
def load_dax_index():
    path = os.path.join(DATA_DIR, "DAX_index.csv")
    return np.genfromtxt(path, usecols=(1,), delimiter=",", skip_header=1)


@st.cache_data
def load_dax_companies():
    path = os.path.join(DATA_DIR, "DAX_companies.csv")
    return np.genfromtxt(path, usecols=(1, 2, 3, 4, 5), delimiter=",", skip_header=1)


def get_log_returns(data):
    return np.diff(np.log(data))
