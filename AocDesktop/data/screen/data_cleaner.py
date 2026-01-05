
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def clean_dataset(df, numeric_columns):
    original_len = len(df)
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    
    removed_count = original_len - len(cleaned_df)
    print(f"Removed {removed_count} outliers from dataset")
    print(f"Original size: {original_len}, Cleaned size: {len(cleaned_df)}")
    
    return cleaned_df

def load_and_clean_data(filepath):
    try:
        df = pd.read_csv(filepath)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cleaned_df = clean_dataset(df, numeric_cols)
        return cleaned_df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    })
    
    sample_data.loc[::100, 'A'] = 500
    sample_data.loc[::50, 'B'] = 300
    
    cleaned = clean_dataset(sample_data, ['A', 'B', 'C'])
    print(f"Sample cleaned - Remaining rows: {len(cleaned)}")