
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import pandas as pd

def remove_duplicates(dataframe, subset=None, keep='first'):
    """
    Remove duplicate rows from a pandas DataFrame.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.
        subset (list, optional): Column labels to consider for identifying duplicates.
        keep (str, optional): Which duplicates to keep. Options: 'first', 'last', False.

    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    if subset is None:
        subset = dataframe.columns.tolist()
    
    cleaned_df = dataframe.drop_duplicates(subset=subset, keep=keep)
    return cleaned_df

def clean_numeric_column(dataframe, column_name, fill_method='mean'):
    """
    Clean a numeric column by handling missing values.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.
        column_name (str): Name of the column to clean.
        fill_method (str): Method to fill missing values ('mean', 'median', 'zero').

    Returns:
        pd.DataFrame: DataFrame with cleaned column.
    """
    if column_name not in dataframe.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    if fill_method == 'mean':
        fill_value = dataframe[column_name].mean()
    elif fill_method == 'median':
        fill_value = dataframe[column_name].median()
    elif fill_method == 'zero':
        fill_value = 0
    else:
        raise ValueError("fill_method must be 'mean', 'median', or 'zero'")
    
    dataframe[column_name] = dataframe[column_name].fillna(fill_value)
    return dataframe

def validate_dataframe(dataframe, required_columns=None):
    """
    Validate DataFrame structure and content.

    Args:
        dataframe (pd.DataFrame): The DataFrame to validate.
        required_columns (list, optional): List of columns that must be present.

    Returns:
        tuple: (is_valid, error_message)
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in dataframe.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    if dataframe.empty:
        return False, "DataFrame is empty"
    
    return True, "DataFrame is valid"