
import pandas as pd
import re

def clean_dataframe(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        column_mapping (dict, optional): Dictionary mapping old column names to new ones
        drop_duplicates (bool): Whether to remove duplicate rows
        normalize_text (bool): Whether to normalize text columns (strip, lower case)
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    # Rename columns if mapping provided
    if column_mapping:
        cleaned_df = cleaned_df.rename(columns=column_mapping)
    
    # Remove duplicate rows
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    
    # Normalize text columns
    if normalize_text:
        text_columns = cleaned_df.select_dtypes(include=['object']).columns
        for col in text_columns:
            cleaned_df[col] = cleaned_df[col].astype(str).str.strip().str.lower()
    
    return cleaned_df

def validate_email(email_series):
    """
    Validate email addresses in a pandas Series.
    
    Args:
        email_series (pd.Series): Series containing email addresses
    
    Returns:
        pd.Series: Boolean series indicating valid emails
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return email_series.str.match(pattern, na=False)

def remove_special_characters(text_series, keep_chars=r'[^a-zA-Z0-9\s]'):
    """
    Remove special characters from text data.
    
    Args:
        text_series (pd.Series): Series containing text data
        keep_chars (str): Regex pattern of characters to keep
    
    Returns:
        pd.Series: Cleaned text series
    """
    return text_series.str.replace(keep_chars, '', regex=True)
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
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

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    If no columns specified, clean all numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[col]):
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{col}'")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    bool: True if validation passes
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    return True

def main():
    """
    Example usage of data cleaning functions.
    """
    np.random.seed(42)
    
    data = {
        'id': range(100),
        'value': np.random.normal(100, 15, 100),
        'score': np.random.uniform(0, 1, 100)
    }
    
    df = pd.DataFrame(data)
    
    df.loc[10, 'value'] = 500
    df.loc[20, 'value'] = -200
    df.loc[30, 'score'] = 5.0
    
    print("Original DataFrame shape:", df.shape)
    
    try:
        validate_dataframe(df, ['id', 'value', 'score'])
        
        cleaned_df = clean_numeric_data(df, ['value', 'score'])
        
        print("Cleaned DataFrame shape:", cleaned_df.shape)
        print("Data cleaning completed successfully")
        
        return cleaned_df
        
    except Exception as e:
        print(f"Error during data cleaning: {e}")
        return None

if __name__ == "__main__":
    cleaned_data = main()import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    outliers_removed = len(data) - len(filtered_data)
    
    return filtered_data, outliers_removed

def z_score_normalize(data, column):
    """
    Normalize data using z-score normalization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column]
    
    normalized = (data[column] - mean_val) / std_val
    return normalized

def min_max_normalize(data, column, new_min=0, new_max=1):
    """
    Normalize data using min-max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column]
    
    normalized = ((data[column] - min_val) / (max_val - min_val)) * (new_max - new_min) + new_min
    return normalized

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in specified columns
    """
    if columns is None:
        columns = data.columns
    
    data_copy = data.copy()
    
    for col in columns:
        if col not in data.columns:
            continue
            
        if data[col].isnull().any():
            if strategy == 'mean':
                fill_value = data[col].mean()
            elif strategy == 'median':
                fill_value = data[col].median()
            elif strategy == 'mode':
                fill_value = data[col].mode()[0]
            elif strategy == 'drop':
                data_copy = data_copy.dropna(subset=[col])
                continue
            else:
                raise ValueError("Strategy must be 'mean', 'median', 'mode', or 'drop'")
            
            data_copy[col] = data_copy[col].fillna(fill_value)
    
    return data_copy

def clean_dataset(data, config):
    """
    Main function to clean dataset based on configuration
    """
    cleaned_data = data.copy()
    
    if 'missing_values' in config:
        cleaned_data = handle_missing_values(
            cleaned_data,
            strategy=config['missing_values'].get('strategy', 'mean'),
            columns=config['missing_values'].get('columns')
        )
    
    if 'outliers' in config:
        for col_config in config['outliers']:
            column = col_config['column']
            method = col_config.get('method', 'iqr')
            
            if method == 'iqr':
                factor = col_config.get('factor', 1.5)
                cleaned_data, removed = remove_outliers_iqr(cleaned_data, column, factor)
    
    if 'normalization' in config:
        for norm_config in config['normalization']:
            column = norm_config['column']
            method = norm_config.get('method', 'zscore')
            
            if method == 'zscore':
                cleaned_data[f'{column}_normalized'] = z_score_normalize(cleaned_data, column)
            elif method == 'minmax':
                new_min = norm_config.get('new_min', 0)
                new_max = norm_config.get('new_max', 1)
                cleaned_data[f'{column}_normalized'] = min_max_normalize(
                    cleaned_data, column, new_min, new_max
                )
    
    return cleaned_data