import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by handling duplicates and missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    drop_duplicates (bool): Whether to drop duplicate rows
    fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    # Remove duplicate rows
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Handle missing values
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
            elif fill_missing == 'median':
                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None, inplace=True)
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of column names that must be present
    min_rows (int): Minimum number of rows required
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Data validation passed"

# Example usage
if __name__ == "__main__":
    # Create sample data with duplicates and missing values
    sample_data = {
        'id': [1, 2, 2, 3, 4],
        'value': [10.5, None, 15.2, 8.7, None],
        'category': ['A', 'B', 'B', 'A', 'C']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nDataFrame info:")
    print(df.info())
    
    # Clean the data
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    # Validate the cleaned data
    is_valid, message = validate_data(cleaned, required_columns=['id', 'value'], min_rows=2)
    print(f"\nValidation result: {is_valid}")
    print(f"Validation message: {message}")
import pandas as pd
import numpy as np

def clean_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): 'mean', 'median', 'mode', or 'drop'
    columns (list): Specific columns to clean, None for all columns
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        columns = df.columns
    
    df_clean = df.copy()
    
    for col in columns:
        if df[col].isnull().any():
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mode':
                fill_value = df[col].mode()[0]
            elif strategy == 'drop':
                df_clean = df_clean.dropna(subset=[col])
                continue
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            df_clean[col] = df[col].fillna(fill_value)
    
    return df_clean

def remove_outliers(df, method='iqr', threshold=1.5, columns=None):
    """
    Remove outliers from DataFrame using specified method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    method (str): 'iqr' or 'zscore'
    threshold (float): Threshold for outlier detection
    columns (list): Specific columns to check, None for all numeric columns
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    
    for col in columns:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
        elif method == 'zscore':
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            mask = z_scores <= threshold
        else:
            raise ValueError(f"Unknown method: {method}")
        
        df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def standardize_data(df, columns=None):
    """
    Standardize numeric columns to have mean=0 and std=1.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): Specific columns to standardize, None for all numeric columns
    
    Returns:
    pd.DataFrame: Standardized DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_std = df.copy()
    
    for col in columns:
        mean_val = df[col].mean()
        std_val = df[col].std()
        if std_val > 0:
            df_std[col] = (df[col] - mean_val) / std_val
    
    return df_std

def normalize_data(df, columns=None, range_min=0, range_max=1):
    """
    Normalize numeric columns to specified range.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): Specific columns to normalize, None for all numeric columns
    range_min (float): Minimum value of normalized range
    range_max (float): Maximum value of normalized range
    
    Returns:
    pd.DataFrame: Normalized DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_norm = df.copy()
    
    for col in columns:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val > min_val:
            df_norm[col] = (df[col] - min_val) / (max_val - min_val) * (range_max - range_min) + range_min
    
    return df_norm

def validate_data(df, rules):
    """
    Validate DataFrame against specified rules.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    rules (dict): Dictionary of validation rules
    
    Returns:
    dict: Validation results
    """
    results = {
        'passed': True,
        'errors': [],
        'warnings': []
    }
    
    for col, rule in rules.items():
        if col in df.columns:
            if 'min' in rule and df[col].min() < rule['min']:
                results['passed'] = False
                results['errors'].append(f"Column {col}: values below minimum {rule['min']}")
            
            if 'max' in rule and df[col].max() > rule['max']:
                results['passed'] = False
                results['errors'].append(f"Column {col}: values above maximum {rule['max']}")
            
            if 'allowed_values' in rule:
                invalid_values = df[~df[col].isin(rule['allowed_values'])][col].unique()
                if len(invalid_values) > 0:
                    results['passed'] = False
                    results['errors'].append(f"Column {col}: invalid values {invalid_values}")
            
            if 'not_null' in rule and rule['not_null'] and df[col].isnull().any():
                results['warnings'].append(f"Column {col}: contains null values")
    
    return results

def get_data_summary(df):
    """
    Generate comprehensive summary of DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    
    Returns:
    dict: Summary statistics
    """
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_stats': {},
        'categorical_stats': {}
    }
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        summary['numeric_stats'][col] = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'q1': df[col].quantile(0.25),
            'q3': df[col].quantile(0.75)
        }
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        summary['categorical_stats'][col] = {
            'unique_count': df[col].nunique(),
            'top_value': df[col].mode()[0] if not df[col].mode().empty else None,
            'top_count': df[col].value_counts().iloc[0] if not df[col].value_counts().empty else 0
        }
    
    return summary