import pandas as pd

def clean_dataframe(df, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.

    Args:
        df (pd.DataFrame): The input DataFrame to clean.
        drop_duplicates (bool): If True, remove duplicate rows. Default is True.
        fill_missing (str, dict, or None): Method to fill missing values.
            If None, missing values are not filled.
            If 'mean', fill with column mean (numeric only).
            If 'median', fill with column median (numeric only).
            If 'mode', fill with column mode.
            If dict, fill with specified values per column.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    df_clean = df.copy()

    if drop_duplicates:
        df_clean = df_clean.drop_duplicates()

    if fill_missing is not None:
        if fill_missing == 'mean':
            df_clean = df_clean.fillna(df_clean.mean(numeric_only=True))
        elif fill_missing == 'median':
            df_clean = df_clean.fillna(df_clean.median(numeric_only=True))
        elif fill_missing == 'mode':
            df_clean = df_clean.fillna(df_clean.mode().iloc[0])
        elif isinstance(fill_missing, dict):
            df_clean = df_clean.fillna(fill_missing)
        else:
            raise ValueError("Unsupported fill_missing method. Use 'mean', 'median', 'mode', or a dict.")

    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame for required columns and basic integrity.

    Args:
        df (pd.DataFrame): The DataFrame to validate.
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

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, None, 5],
        'B': [10, None, 30, 40, 50],
        'C': ['x', 'y', 'y', 'z', None]
    }
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)

    cleaned_df = clean_dataframe(df, fill_missing='mean')
    print("\nCleaned DataFrame (filled with mean for numeric columns):")
    print(cleaned_df)

    is_valid = validate_dataframe(cleaned_df, required_columns=['A', 'B'])
    print(f"\nDataFrame validation passed: {is_valid}")