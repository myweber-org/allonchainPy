
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
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
    
    return filtered_df.reset_index(drop=True)

def calculate_summary_stats(df, column):
    """
    Calculate summary statistics for a column.
    
    Parameters:
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
        'count': len(df[column]),
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
    print("Original summary statistics:")
    print(calculate_summary_stats(df, 'value'))
    
    cleaned_df = remove_outliers_iqr(df, 'value')
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("Cleaned summary statistics:")
    print(calculate_summary_stats(cleaned_df, 'value'))
    
    return cleaned_df

if __name__ == "__main__":
    cleaned_data = example_usage()import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        subset (list, optional): Column names to consider for duplicates
        keep (str, optional): Which duplicates to keep ('first', 'last', False)
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(df) - len(cleaned_df)
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df.reset_index(drop=True)

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list, optional): List of required column names
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame")
        return False
    
    if df.empty:
        print("Warning: DataFrame is empty")
        return True
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return False
    
    return True

def clean_numeric_columns(df, columns=None):
    """
    Clean numeric columns by converting to appropriate types and handling errors.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): Specific columns to clean
    
    Returns:
        pd.DataFrame: DataFrame with cleaned numeric columns
    """
    if columns is None:
        columns = df.select_dtypes(include=['object']).columns
    
    for col in columns:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception as e:
                print(f"Warning: Could not convert column '{col}' to numeric: {e}")
    
    return df

def get_data_summary(df):
    """
    Generate a summary of the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
    
    Returns:
        dict: Summary statistics
    """
    summary = {
        'rows': len(df),
        'columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'column_types': df.dtypes.to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum()
    }
    
    return summary
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a dataset using the Interquartile Range (IQR) method.
    
    Parameters:
    data (np.ndarray): Input data array.
    column (int): Index of the column to process.
    
    Returns:
    np.ndarray: Data with outliers removed.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    
    if column >= data.shape[1]:
        raise IndexError("Column index out of bounds")
    
    col_data = data[:, column]
    q1 = np.percentile(col_data, 25)
    q3 = np.percentile(col_data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    mask = (col_data >= lower_bound) & (col_data <= upper_bound)
    return data[mask]

def calculate_statistics(data):
    """
    Calculate basic statistics for the cleaned data.
    
    Parameters:
    data (np.ndarray): Input data array.
    
    Returns:
    dict: Dictionary containing mean, median, and standard deviation.
    """
    if data.size == 0:
        return {"mean": np.nan, "median": np.nan, "std": np.nan}
    
    return {
        "mean": np.mean(data),
        "median": np.median(data),
        "std": np.std(data)
    }

def process_dataset(data, column):
    """
    Main function to process dataset by removing outliers and calculating statistics.
    
    Parameters:
    data (np.ndarray): Input data array.
    column (int): Index of the column to process.
    
    Returns:
    tuple: Cleaned data and statistics dictionary.
    """
    cleaned_data = remove_outliers_iqr(data, column)
    stats = calculate_statistics(cleaned_data[:, column])
    return cleaned_data, stats

if __name__ == "__main__":
    sample_data = np.array([
        [1.2, 10.5],
        [2.3, 12.1],
        [3.1, 100.0],
        [4.2, 13.2],
        [5.5, 14.8],
        [6.7, 15.3],
        [7.9, 200.0],
        [8.4, 16.7]
    ])
    
    result, statistics = process_dataset(sample_data, column=1)
    print("Cleaned data shape:", result.shape)
    print("Statistics:", statistics)