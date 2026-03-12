
import pandas as pd

def filter_and_clean(df, column_name, valid_values=None, drop_na=True):
    """
    Filters a DataFrame based on a column's values and optionally drops NA rows.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the column to filter on.
        valid_values (list, optional): A list of valid values to keep.
                                       If None, only checks for NA.
        drop_na (bool): If True, drops rows where the column is NA.

    Returns:
        pd.DataFrame: The filtered and cleaned DataFrame.
    """
    filtered_df = df.copy()

    if drop_na:
        filtered_df = filtered_df.dropna(subset=[column_name])

    if valid_values is not None:
        filtered_df = filtered_df[filtered_df[column_name].isin(valid_values)]

    return filtered_df.reset_index(drop=True)