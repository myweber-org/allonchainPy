import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range (IQR) method.
    
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
    
    return filtered_df

def clean_dataset(df, numeric_columns=None):
    """
    Clean a dataset by removing outliers from all numeric columns.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    numeric_columns (list): List of numeric column names. If None, uses all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame with outliers removed.
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in df.columns:
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, col)
            except Exception as e:
                print(f"Warning: Could not process column '{col}': {e}")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[0, 'A'] = 1000
    df.loc[1, 'B'] = 500
    
    print("Original DataFrame shape:", df.shape)
    print("Original statistics:")
    print(df.describe())
    
    cleaned_df = clean_dataset(df)
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("Cleaned statistics:")
    print(cleaned_df.describe())import pandas as pd
import numpy as np

def clean_dataframe(df, drop_na=True, rename_columns=True):
    """
    Clean a pandas DataFrame by handling missing values and standardizing column names.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_na (bool): If True, drop rows with any null values
        rename_columns (bool): If True, rename columns to lowercase with underscores
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if drop_na:
        df_clean = df_clean.dropna()
    
    if rename_columns:
        df_clean.columns = (
            df_clean.columns
            .str.lower()
            .str.replace(r'[^a-z0-9]+', '_', regex=True)
            .str.strip('_')
        )
    
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        dict: Dictionary with validation results
    """
    validation_result = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'summary': {
            'rows': len(df),
            'columns': len(df.columns),
            'null_count': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum()
        }
    }
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f'Missing columns: {missing_cols}')
    
    if df.empty:
        validation_result['is_valid'] = False
        validation_result['errors'].append('DataFrame is empty')
    
    if df.isnull().any().any():
        validation_result['warnings'].append('DataFrame contains null values')
    
    if df.duplicated().any():
        validation_result['warnings'].append('DataFrame contains duplicate rows')
    
    return validation_result

def sample_data():
    """
    Create sample DataFrame for testing.
    
    Returns:
        pd.DataFrame: Sample DataFrame with mixed data
    """
    data = {
        'Product Name': ['Widget A', 'Widget B', None, 'Widget C'],
        'Price ($)': [19.99, 29.99, 'invalid', 39.99],
        'Quantity': [100, 200, 150, None],
        'Category': ['Electronics', 'Electronics', 'Home', 'Electronics']
    }
    
    return pd.DataFrame(data)