from util.load_packages import st, np, pd



# --- 1. Data Loading ---
class DataLoader:
    def __init__(self, path: str):
        self.path = path
    
    def load_csv(self) -> pd.DataFrame:
        df = pd.read_csv(self.source)
        return df




class DataTrafo:
    def __init__(self):
        pass



class RiskModel:
    def __init__(self):
        pass




loader_obj = DataLoader("data/DAX_index.csv")
raw_df = loader_obj.load_csv()

st.dataframe(raw_df)