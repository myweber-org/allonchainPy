import pandas as pd
import numpy as np

def clean_missing_data(df, strategy='mean', columns=None):
    """
    Clean missing data in a pandas DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    strategy (str): Strategy for handling missing values.
                    Options: 'mean', 'median', 'mode', 'drop', 'fill'.
    columns (list): List of columns to apply cleaning. If None, applies to all numeric columns.

    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    if df.empty:
        return df

    df_cleaned = df.copy()

    if columns is None:
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols

    for col in columns:
        if col not in df_cleaned.columns:
            continue

        if strategy == 'mean':
            fill_value = df_cleaned[col].mean()
        elif strategy == 'median':
            fill_value = df_cleaned[col].median()
        elif strategy == 'mode':
            fill_value = df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else np.nan
        elif strategy == 'drop':
            df_cleaned = df_cleaned.dropna(subset=[col])
            continue
        elif strategy == 'fill':
            fill_value = 0
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

        df_cleaned[col] = df_cleaned[col].fillna(fill_value)

    return df_cleaned

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.

    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.

    Returns:
    bool: True if validation passes, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        return False

    if df.empty:
        return False

    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False

    return True

def load_and_clean_csv(file_path, **kwargs):
    """
    Load CSV file and clean missing data.

    Parameters:
    file_path (str): Path to CSV file.
    **kwargs: Additional arguments passed to clean_missing_data.

    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        if validate_dataframe(df):
            return clean_missing_data(df, **kwargs)
        else:
            return pd.DataFrame()
    except Exception as e:
        print(f"Error loading file: {e}")
        return pd.DataFrame()