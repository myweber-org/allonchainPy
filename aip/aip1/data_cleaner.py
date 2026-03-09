
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_na=None):
    """
    Clean a pandas DataFrame by handling missing values and removing duplicates.

    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): If True, remove duplicate rows.
    fill_na (optional): Value to fill missing entries with. If None, rows with nulls are dropped.

    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()

    if fill_na is not None:
        cleaned_df = cleaned_df.fillna(fill_na)
    else:
        cleaned_df = cleaned_df.dropna()

    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()

    return cleaned_df

def validate_data(df, required_columns=None):
    """
    Validate the DataFrame structure and content.

    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.

    Returns:
    bool: True if validation passes, False otherwise.
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return False

    if df.empty:
        print("DataFrame is empty.")
        return False

    return True