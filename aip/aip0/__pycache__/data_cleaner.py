
import pandas as pd
import numpy as np

def clean_dataset(df):
    """
    Remove duplicate rows and standardize column names.
    """
    # Remove duplicates
    df_cleaned = df.drop_duplicates()
    
    # Standardize column names
    df_cleaned.columns = df_cleaned.columns.str.strip().str.lower().str.replace(' ', '_')
    
    return df_cleaned

def handle_missing_values(df, strategy='mean'):
    """
    Handle missing values using specified strategy.
    """
    if strategy == 'mean':
        return df.fillna(df.mean())
    elif strategy == 'median':
        return df.fillna(df.median())
    elif strategy == 'mode':
        return df.fillna(df.mode().iloc[0])
    else:
        return df.fillna(0)

def remove_outliers(df, column, threshold=3):
    """
    Remove outliers using z-score method.
    """
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    return df[z_scores < threshold]

def main():
    # Example usage
    data = pd.DataFrame({
        'ID': [1, 2, 3, 4, 5, 5],
        'Value': [10, 20, np.nan, 40, 50, 50],
        'Category': ['A', 'B', 'C', 'D', 'E', 'E']
    })
    
    print("Original Data:")
    print(data)
    
    cleaned_data = clean_dataset(data)
    print("\nAfter cleaning:")
    print(cleaned_data)
    
    filled_data = handle_missing_values(cleaned_data)
    print("\nAfter handling missing values:")
    print(filled_data)

if __name__ == "__main__":
    main()