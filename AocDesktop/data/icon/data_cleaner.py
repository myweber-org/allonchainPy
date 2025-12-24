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
import pandas as pd

def clean_dataset(df, id_column=None):
    """
    Clean a pandas DataFrame by removing duplicates and standardizing column names.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        id_column (str, optional): Column name to use for identifying duplicates.
                                   If None, uses all columns.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Standardize column names: lowercase and replace spaces with underscores
    cleaned_df.columns = cleaned_df.columns.str.lower().str.replace(' ', '_')
    
    # Remove duplicate rows
    if id_column and id_column in cleaned_df.columns:
        cleaned_df = cleaned_df.drop_duplicates(subset=[id_column])
    else:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and required columns.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list, optional): List of required column names.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'Customer ID': [1, 2, 2, 3, 4],
        'First Name': ['John', 'Jane', 'Jane', 'Bob', 'Alice'],
        'Last Name': ['Doe', 'Smith', 'Smith', 'Johnson', 'Brown'],
        'Email': ['john@example.com', 'jane@example.com', 'jane@example.com', 'bob@example.com', 'alice@example.com']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned = clean_dataset(df, id_column='Customer ID')
    print(cleaned)
    
    # Validate the cleaned DataFrame
    is_valid, message = validate_dataframe(cleaned, required_columns=['customer_id', 'email'])
    print(f"\nValidation: {message}")import pandas as pd

def clean_dataframe(df, drop_na=True, rename_columns=True):
    """
    Clean a pandas DataFrame by removing null values and standardizing column names.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    drop_na (bool): Whether to drop rows with null values
    rename_columns (bool): Whether to rename columns to lowercase with underscores
    
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
    
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and required columns.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    if not isinstance(df, pd.DataFrame):
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    return True

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    subset (list): Columns to consider for identifying duplicates
    keep (str): Which duplicates to keep ('first', 'last', False)
    
    Returns:
    pd.DataFrame: DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep=keep)