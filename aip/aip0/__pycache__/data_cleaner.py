import pandas as pd
import numpy as np
from scipy import stats

def load_and_clean_data(filepath):
    """
    Load CSV data, remove outliers using z-score,
    and normalize numeric columns.
    """
    df = pd.read_csv(filepath)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        z_scores = np.abs(stats.zscore(df[col].dropna()))
        df = df[(z_scores < 3) | df[col].isna()]
    
    for col in numeric_cols:
        if df[col].std() > 0:
            df[col] = (df[col] - df[col].mean()) / df[col].std()
    
    return df.reset_index(drop=True)

def save_cleaned_data(df, output_path):
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    cleaned_df = load_and_clean_data("raw_data.csv")
    save_cleaned_data(cleaned_df, "cleaned_data.csv")