import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df: pandas DataFrame to clean
        drop_duplicates: If True, remove duplicate rows
        fill_missing: Strategy for handling missing values. Options:
            - None: drop rows with any missing values
            - 'mean': fill with column mean (numeric only)
            - 'median': fill with column median (numeric only)
            - 'mode': fill with column mode
            - dict: column-specific fill values
    
    Returns:
        Cleaned pandas DataFrame
    """
    cleaned_df = df.copy()
    
    # Remove duplicate rows
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Handle missing values
    if fill_missing is None:
        cleaned_df = cleaned_df.dropna()
    elif fill_missing == 'mean':
        cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
    elif fill_missing == 'median':
        cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
    elif fill_missing == 'mode':
        cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
    elif isinstance(fill_missing, dict):
        cleaned_df = cleaned_df.fillna(fill_missing)
    
    return cleaned_df

def validate_dataset(df, required_columns=None, min_rows=1):
    """
    Validate dataset structure and content.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: List of column names that must be present
        min_rows: Minimum number of rows required
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if len(df) < min_rows:
        return False, f"Dataset must have at least {min_rows} rows"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "Dataset is valid"

def remove_outliers(df, column, method='iqr', threshold=1.5):
    """
    Remove outliers from a specific column using specified method.
    
    Args:
        df: pandas DataFrame
        column: Column name to check for outliers
        method: Outlier detection method ('iqr' or 'zscore')
        threshold: Threshold for outlier detection
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        mask = (df[column] >= lower_bound) & (df[column] <= upper_bound)
    elif method == 'zscore':
        from scipy import stats
        z_scores = stats.zscore(df[column].dropna())
        mask = abs(z_scores) < threshold
        # Align mask with original DataFrame
        mask = pd.Series(mask, index=df[column].dropna().index)
        mask = mask.reindex(df.index).fillna(True)
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")
    
    return df[mask]