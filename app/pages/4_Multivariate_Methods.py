from util.load_packages import st, np, pd



# --- 1. Data Loading ---
class DataLoader:
    def __init__(self, source: str):
        self.source = source
    
    def load_csv(self) -> pd.DataFrame:
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

st.dataframe(raw_df)
