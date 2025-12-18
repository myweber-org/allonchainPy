import numpy as np
import pandas as pd

def remove_outliers_iqr(dataframe, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Args:
        dataframe: pandas DataFrame containing the data
        column: string name of the column to process
    
    Returns:
        pandas DataFrame with outliers removed
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)]
    
    return filtered_df

def calculate_basic_stats(dataframe, column):
    """
    Calculate basic statistics for a specified column.
    
    Args:
        dataframe: pandas DataFrame containing the data
        column: string name of the column to analyze
    
    Returns:
        dictionary containing statistical measures
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': dataframe[column].mean(),
        'median': dataframe[column].median(),
        'std': dataframe[column].std(),
        'min': dataframe[column].min(),
        'max': dataframe[column].max(),
        'count': dataframe[column].count()
    }
    
    return stats

def clean_dataset(dataframe, numeric_columns=None):
    """
    Clean dataset by removing outliers from all numeric columns.
    
    Args:
        dataframe: pandas DataFrame to clean
        numeric_columns: list of column names to process (defaults to all numeric columns)
    
    Returns:
        cleaned pandas DataFrame
    """
    if numeric_columns is None:
        numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = dataframe.copy()
    
    for column in numeric_columns:
        if column in cleaned_df.columns:
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, column)
            except Exception as e:
                print(f"Warning: Could not process column '{column}': {e}")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'temperature': [22, 23, 24, 25, 26, 27, 28, 29, 30, 100],
        'humidity': [45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
        'pressure': [1013, 1012, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1100]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\nOriginal statistics for 'temperature':")
    print(calculate_basic_stats(df, 'temperature'))
    
    cleaned_df = clean_dataset(df)
    print("\nCleaned dataset:")
    print(cleaned_df)
    print("\nCleaned statistics for 'temperature':")
    print(calculate_basic_stats(cleaned_df, 'temperature'))import pandas as pd

def clean_dataset(df, drop_duplicates=True, fillna_method=None, fillna_value=None):
    """
    Clean a pandas DataFrame by handling missing values and duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default True.
    fillna_method (str): Method to fill missing values ('ffill', 'bfill', etc). Default None.
    fillna_value: Value to fill missing values with if fillna_method is None.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if fillna_method is not None:
        cleaned_df = cleaned_df.fillna(method=fillna_method)
    elif fillna_value is not None:
        cleaned_df = cleaned_df.fillna(fillna_value)
    
    # Remove duplicates
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate dataset for required columns and basic integrity.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    return True, "Dataset is valid"

# Example usage
if __name__ == "__main__":
    # Create sample data with missing values and duplicates
    sample_data = {
        'id': [1, 2, 2, 3, 4, 4],
        'name': ['Alice', 'Bob', 'Bob', None, 'Eve', 'Eve'],
        'score': [85, 90, 90, 78, None, None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nDataFrame info:")
    print(df.info())
    
    # Clean the dataset
    cleaned = clean_dataset(df, fillna_value=0)
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    # Validate the cleaned dataset
    is_valid, message = validate_dataset(cleaned, required_columns=['id', 'name', 'score'])
    print(f"\nValidation: {message}")
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Parameters:
    data (DataFrame): The input data.
    column (str): The column name to clean.
    
    Returns:
    DataFrame: Data with outliers removed.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    return filtered_data

def calculate_summary_statistics(data, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    data (DataFrame): The input data.
    column (str): The column name.
    
    Returns:
    dict: Summary statistics.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    stats = {
        'mean': data[column].mean(),
        'median': data[column].median(),
        'std': data[column].std(),
        'min': data[column].min(),
        'max': data[column].max()
    }
    
    return stats