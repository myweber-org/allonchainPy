
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to process
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def clean_dataset(df, numeric_columns=None):
    """
    Clean dataset by removing outliers from all numeric columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        numeric_columns (list, optional): List of numeric columns to clean.
                                         If None, uses all numeric columns.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in df.columns:
            original_len = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_len - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 3, 4, 5, 100],
        'B': [10, 20, 30, 40, 50, 200],
        'C': ['a', 'b', 'c', 'd', 'e', 'f']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_dataset(df, ['A', 'B'])
    print("\nCleaned DataFrame:")
    print(cleaned_df)