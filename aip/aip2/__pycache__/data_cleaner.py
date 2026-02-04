
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas DataFrame column using the IQR method.
    
    Parameters:
    data (pd.DataFrame): The DataFrame containing the data.
    column (str): The column name to clean.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
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
    data (pd.DataFrame): The DataFrame containing the data.
    column (str): The column name to analyze.
    
    Returns:
    dict: Dictionary containing mean, median, and standard deviation.
    """
    stats = {
        'mean': data[column].mean(),
        'median': data[column].median(),
        'std': data[column].std()
    }
    return stats

if __name__ == "__main__":
    import pandas as pd
    
    sample_data = pd.DataFrame({
        'values': [10, 12, 12, 13, 12, 11, 10, 100, 12, 14, 12, 10, 9, 9, 200]
    })
    
    print("Original data:")
    print(sample_data)
    
    cleaned_data = remove_outliers_iqr(sample_data, 'values')
    print("\nCleaned data:")
    print(cleaned_data)
    
    stats = calculate_summary_statistics(cleaned_data, 'values')
    print("\nSummary statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    drop_duplicates (bool): Whether to drop duplicate rows
    fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
    
    return cleaned_df

def validate_data(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing required columns: {missing_cols}")
    
    if df.empty:
        validation_results['warnings'].append("DataFrame is empty")
    
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            validation_results['warnings'].append(f"Column '{col}' contains missing values")
    
    return validation_resultsimport pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    subset (list, optional): Column labels to consider for duplicates.
    keep (str, optional): Which duplicates to keep.
    
    Returns:
    pd.DataFrame: DataFrame with duplicates removed.
    """
    if df.empty:
        return df
        
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    removed_count = len(df) - len(cleaned_df)
    
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df

def clean_numeric_column(df, column_name, fill_method='mean'):
    """
    Clean numeric column by handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column_name (str): Name of column to clean.
    fill_method (str): Method to fill missing values ('mean', 'median', 'zero').
    
    Returns:
    pd.DataFrame: DataFrame with cleaned column.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    if not pd.api.types.is_numeric_dtype(df[column_name]):
        raise TypeError(f"Column '{column_name}' is not numeric")
    
    df_clean = df.copy()
    missing_count = df_clean[column_name].isna().sum()
    
    if missing_count > 0:
        if fill_method == 'mean':
            fill_value = df_clean[column_name].mean()
        elif fill_method == 'median':
            fill_value = df_clean[column_name].median()
        elif fill_method == 'zero':
            fill_value = 0
        else:
            raise ValueError("fill_method must be 'mean', 'median', or 'zero'")
        
        df_clean[column_name] = df_clean[column_name].fillna(fill_value)
        print(f"Filled {missing_count} missing values in '{column_name}' with {fill_method}: {fill_value}")
    
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list, optional): List of required column names.
    
    Returns:
    bool: True if validation passes, False otherwise.
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
    
    print(f"DataFrame validation passed: {len(df)} rows, {len(df.columns)} columns")
    return True

def main():
    """
    Example usage of data cleaning functions.
    """
    sample_data = {
        'id': [1, 2, 2, 3, 4, 4, 5],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', 'David', 'David', 'Eve'],
        'score': [85, 90, 90, None, 78, 78, 92],
        'age': [25, 30, 30, 35, None, 28, 32]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    if validate_dataframe(df, required_columns=['id', 'name']):
        df_clean = remove_duplicates(df, subset=['id', 'name'])
        df_clean = clean_numeric_column(df_clean, 'score', fill_method='mean')
        df_clean = clean_numeric_column(df_clean, 'age', fill_method='median')
        
        print("\nCleaned DataFrame:")
        print(df_clean)

if __name__ == "__main__":
    main()import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df: pandas DataFrame
        subset: column label or sequence of labels to consider for duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def fill_missing_values(df, strategy='mean', columns=None):
    """
    Fill missing values in specified columns using a strategy.
    
    Args:
        df: pandas DataFrame
        strategy: 'mean', 'median', 'mode', or 'constant'
        columns: list of column names to fill (None for all columns)
    
    Returns:
        DataFrame with missing values filled
    """
    if columns is None:
        columns = df.columns
    
    df_filled = df.copy()
    
    for col in columns:
        if col in df.columns and df[col].isnull().any():
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mode':
                fill_value = df[col].mode()[0]
            elif strategy == 'constant':
                fill_value = 0
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            df_filled[col] = df[col].fillna(fill_value)
    
    return df_filled

def normalize_column(df, column, method='minmax'):
    """
    Normalize a column using specified method.
    
    Args:
        df: pandas DataFrame
        column: column name to normalize
        method: 'minmax' or 'zscore'
    
    Returns:
        DataFrame with normalized column
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_normalized = df.copy()
    
    if method == 'minmax':
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val != min_val:
            df_normalized[column] = (df[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = df[column].mean()
        std_val = df[column].std()
        if std_val != 0:
            df_normalized[column] = (df[column] - mean_val) / std_val
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return df_normalized

def detect_outliers(df, column, method='iqr', threshold=1.5):
    """
    Detect outliers in a column.
    
    Args:
        df: pandas DataFrame
        column: column name to check for outliers
        method: 'iqr' or 'zscore'
        threshold: threshold value for outlier detection
    
    Returns:
        Boolean Series indicating outliers
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (df[column] < lower_bound) | (df[column] > upper_bound)
    
    elif method == 'zscore':
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        return z_scores > threshold
    
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")

def clean_dataframe(df, operations):
    """
    Apply multiple cleaning operations to a DataFrame.
    
    Args:
        df: pandas DataFrame
        operations: list of dictionaries specifying cleaning operations
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    for op in operations:
        if op['type'] == 'remove_duplicates':
            cleaned_df = remove_duplicates(cleaned_df, op.get('subset'))
        
        elif op['type'] == 'fill_missing':
            cleaned_df = fill_missing_values(
                cleaned_df, 
                op.get('strategy', 'mean'),
                op.get('columns')
            )
        
        elif op['type'] == 'normalize':
            cleaned_df = normalize_column(
                cleaned_df,
                op['column'],
                op.get('method', 'minmax')
            )
    
    return cleaned_df