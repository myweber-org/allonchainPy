import pandas as pd
import numpy as np

def load_data(filepath):
    """Load dataset from CSV file."""
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return None

def remove_outliers_iqr(df, column):
    """Remove outliers using IQR method for specified column."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    removed_count = len(df) - len(filtered_df)
    print(f"Removed {removed_count} outliers from column '{column}'")
    return filtered_df

def normalize_column(df, column):
    """Normalize column values to range [0, 1]."""
    if column in df.columns:
        min_val = df[column].min()
        max_val = df[column].max()
        
        if max_val != min_val:
            df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
            print(f"Column '{column}' normalized successfully.")
        else:
            print(f"Warning: Column '{column}' has constant values.")
    else:
        print(f"Error: Column '{column}' not found in dataframe.")
    return df

def clean_dataset(filepath, numeric_columns):
    """Main function to clean dataset."""
    df = load_data(filepath)
    
    if df is None:
        return None
    
    print("\nInitial data info:")
    print(df.info())
    
    original_shape = df.shape
    
    for column in numeric_columns:
        if column in df.columns:
            df = remove_outliers_iqr(df, column)
    
    for column in numeric_columns:
        if column in df.columns:
            df = normalize_column(df, column)
    
    print(f"\nCleaning complete.")
    print(f"Original shape: {original_shape}")
    print(f"Final shape: {df.shape}")
    
    return df

if __name__ == "__main__":
    data_file = "sample_data.csv"
    numeric_cols = ['age', 'income', 'score']
    
    cleaned_data = clean_dataset(data_file, numeric_cols)
    
    if cleaned_data is not None:
        output_file = "cleaned_data.csv"
        cleaned_data.to_csv(output_file, index=False)
        print(f"\nCleaned data saved to '{output_file}'")