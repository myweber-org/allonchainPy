
import pandas as pd
import numpy as np

def clean_dataframe(df, drop_duplicates=True, fill_missing='mean'):
    """
    Cleans a pandas DataFrame by removing duplicates and handling missing values.
    """
    cleaned_df = df.copy()

    if drop_duplicates:
        initial_rows = cleaned_df.shape[0]
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - cleaned_df.shape[0]
        print(f"Removed {removed} duplicate row(s).")

    if cleaned_df.isnull().sum().any():
        print("Handling missing values...")
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        non_numeric_cols = cleaned_df.select_dtypes(exclude=[np.number]).columns

        if fill_missing == 'mean' and len(numeric_cols) > 0:
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].mean())
        elif fill_missing == 'median' and len(numeric_cols) > 0:
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].median())
        elif fill_missing == 'zero' and len(numeric_cols) > 0:
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(0)

        for col in non_numeric_cols:
            if cleaned_df[col].isnull().any():
                cleaned_df[col] = cleaned_df[col].fillna('Unknown')

    print(f"Data cleaning complete. Final shape: {cleaned_df.shape}")
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, None, 6],
        'B': [5.1, None, 5.1, 8.2, 9.0, 8.2],
        'C': ['x', 'y', 'x', None, 'z', 'y']
    }
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaning DataFrame...")
    result_df = clean_dataframe(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(result_df)