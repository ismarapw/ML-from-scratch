import pandas as pd

def load_df_from_csv(path:str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df
