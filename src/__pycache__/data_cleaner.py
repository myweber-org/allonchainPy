
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a specified column in a DataFrame using the IQR method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed from the specified column.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df.reset_index(drop=True)

def calculate_basic_stats(df, column):
    """
    Calculate basic statistics for a column after outlier removal.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to analyze.
    
    Returns:
    dict: Dictionary containing count, mean, std, min, and max.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'count': df[column].count(),
        'mean': df[column].mean(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max()
    }
    
    return stats

def process_numerical_data(file_path, column_name):
    """
    Main function to load data, remove outliers, and return cleaned data with statistics.
    
    Parameters:
    file_path (str): Path to the CSV file.
    column_name (str): Column to process.
    
    Returns:
    tuple: (cleaned DataFrame, statistics dictionary)
    """
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at path: {file_path}")
    
    cleaned_data = remove_outliers_iqr(data, column_name)
    statistics = calculate_basic_stats(cleaned_data, column_name)
    
    return cleaned_data, statistics

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'values': np.concatenate([np.random.normal(100, 15, 95), np.array([500, -300, 1000])])
    })
    
    cleaned_df, stats = process_numerical_data('sample_data.csv', 'values')
    
    print("Original data shape:", sample_data.shape)
    print("Cleaned data shape:", cleaned_df.shape)
    print("\nStatistics after cleaning:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")