
import pandas as pd
import numpy as np

def clean_data(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    drop_duplicates (bool): If True, remove duplicate rows.
    fill_missing (str): Strategy to fill missing values.
                        Options: 'mean', 'median', 'mode', or 'drop'.

    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()

    if drop_duplicates:
        df_clean = df_clean.drop_duplicates()

    if fill_missing == 'drop':
        df_clean = df_clean.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                fill_value = df_clean[col].mean()
            else:
                fill_value = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(fill_value)
    elif fill_missing == 'mode':
        for col in df_clean.columns:
            mode_value = df_clean[col].mode()
            if not mode_value.empty:
                df_clean[col] = df_clean[col].fillna(mode_value.iloc[0])

    return df_clean

def main():
    data = {
        'A': [1, 2, 2, np.nan, 5],
        'B': [10, np.nan, 10, 40, 50],
        'C': ['x', 'y', 'x', 'y', 'z']
    }
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)

    df_cleaned = clean_data(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(df_cleaned)

if __name__ == "__main__":
    main()