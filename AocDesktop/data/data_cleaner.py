import pandas as pd
import numpy as np

def clean_csv_data(file_path, strategy='mean', columns=None):
    """
    Clean a CSV file by handling missing values.
    
    Args:
        file_path (str): Path to the CSV file.
        strategy (str): Strategy for filling missing values.
                        Options: 'mean', 'median', 'mode', 'drop'.
        columns (list): List of column names to apply cleaning to.
                        If None, applies to all numeric columns.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if strategy == 'mean':
            fill_value = df[col].mean()
        elif strategy == 'median':
            fill_value = df[col].median()
        elif strategy == 'mode':
            fill_value = df[col].mode()[0] if not df[col].mode().empty else np.nan
        elif strategy == 'drop':
            df = df.dropna(subset=[col])
            continue
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        df[col] = df[col].fillna(fill_value)
    
    return df

def save_cleaned_data(df, output_path):
    """
    Save cleaned DataFrame to a new CSV file.
    
    Args:
        df (pd.DataFrame): Cleaned DataFrame.
        output_path (str): Path to save the cleaned CSV file.
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = clean_csv_data(input_file, strategy='median')
    save_cleaned_data(cleaned_df, output_file)
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to clean.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
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
    Calculate summary statistics for a DataFrame column.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to analyze.
    
    Returns:
    dict: Dictionary containing summary statistics.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

def example_usage():
    """
    Example usage of the data cleaning functions.
    """
    np.random.seed(42)
    data = {
        'id': range(100),
        'value': np.random.normal(100, 15, 100)
    }
    
    df = pd.DataFrame(data)
    
    df.loc[10, 'value'] = 500
    df.loc[20, 'value'] = -100
    
    print("Original DataFrame shape:", df.shape)
    print("\nOriginal summary statistics:")
    original_stats = calculate_summary_statistics(df, 'value')
    for key, value in original_stats.items():
        print(f"{key}: {value:.2f}")
    
    cleaned_df = remove_outliers_iqr(df, 'value')
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("\nCleaned summary statistics:")
    cleaned_stats = calculate_summary_statistics(cleaned_df, 'value')
    for key, value in cleaned_stats.items():
        print(f"{key}: {value:.2f}")
    
    removed_count = df.shape[0] - cleaned_df.shape[0]
    print(f"\nRemoved {removed_count} outliers")
    
    return cleaned_df

if __name__ == "__main__":
    cleaned_data = example_usage()