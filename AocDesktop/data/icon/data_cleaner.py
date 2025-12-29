import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to process
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def clean_dataset(df, numeric_columns):
    """
    Clean dataset by removing outliers from multiple numeric columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        numeric_columns (list): List of column names to clean
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in cleaned_df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_df

def main():
    # Example usage
    data = {
        'id': range(1, 101),
        'value': np.random.randn(100) * 100 + 50,
        'score': np.random.randn(100) * 20 + 75
    }
    
    df = pd.DataFrame(data)
    print(f"Original dataset shape: {df.shape}")
    
    # Add some outliers
    df.loc[5, 'value'] = 1000
    df.loc[10, 'score'] = -200
    
    numeric_cols = ['value', 'score']
    cleaned_df = clean_dataset(df, numeric_cols)
    
    print(f"Cleaned dataset shape: {cleaned_df.shape}")
    print(f"Removed {len(df) - len(cleaned_df)} total outliers")
    
    return cleaned_df

if __name__ == "__main__":
    cleaned_data = main()