import pandas as pd
import numpy as np
from scipy import stats

def load_and_clean_data(filepath):
    """
    Load a CSV file and perform basic data cleaning operations.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

    # Remove duplicate rows
    initial_count = len(df)
    df.drop_duplicates(inplace=True)
    duplicates_removed = initial_count - len(df)
    print(f"Removed {duplicates_removed} duplicate rows.")

    # Handle missing values: drop rows where all values are NaN
    df.dropna(how='all', inplace=True)
    # For numeric columns, fill missing values with column median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    # Remove outliers using Z-score for numeric columns
    z_scores = np.abs(stats.zscore(df[numeric_cols]))
    outlier_mask = (z_scores < 3).all(axis=1)  # Keep rows where all Z-scores < 3
    df_clean = df[outlier_mask].copy()
    outliers_removed = len(df) - len(df_clean)
    print(f"Removed {outliers_removed} rows based on Z-score outlier detection.")

    # Normalize numeric columns to range [0, 1]
    for col in numeric_cols:
        if df_clean[col].max() != df_clean[col].min():  # Avoid division by zero
            df_clean[col] = (df_clean[col] - df_clean[col].min()) / (df_clean[col].max() - df_clean[col].min())

    print(f"Final cleaned data shape: {df_clean.shape}")
    return df_clean

def save_cleaned_data(df, output_path):
    """
    Save the cleaned DataFrame to a CSV file.
    """
    if df is not None:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")
    else:
        print("No data to save.")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"

    cleaned_df = load_and_clean_data(input_file)
    save_cleaned_data(cleaned_df, output_file)import pandas as pd
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
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to analyze
    
    Returns:
        dict: Dictionary containing summary statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

def process_dataframe(df, column):
    """
    Complete data processing pipeline: remove outliers and return statistics.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to process
    
    Returns:
        tuple: (cleaned_df, original_stats, cleaned_stats)
    """
    original_stats = calculate_summary_statistics(df, column)
    cleaned_df = remove_outliers_iqr(df, column)
    cleaned_stats = calculate_summary_statistics(cleaned_df, column)
    
    return cleaned_df, original_stats, cleaned_stats