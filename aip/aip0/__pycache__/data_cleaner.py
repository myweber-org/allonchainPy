
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
import pandas as pd
import numpy as np
from typing import List, Optional

def clean_dataframe(df: pd.DataFrame, 
                    drop_duplicates: bool = True,
                    fill_missing: Optional[str] = 'mean',
                    columns_to_standardize: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Clean a pandas DataFrame by removing duplicates, handling missing values,
    and standardizing specified columns.
    """
    df_clean = df.copy()
    
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates()
    
    if fill_missing:
        if fill_missing == 'mean':
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
        elif fill_missing == 'median':
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
        elif fill_missing == 'mode':
            for col in df_clean.columns:
                mode_val = df_clean[col].mode()
                if not mode_val.empty:
                    df_clean[col] = df_clean[col].fillna(mode_val.iloc[0])
    
    if columns_to_standardize:
        for col in columns_to_standardize:
            if col in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[col]):
                mean = df_clean[col].mean()
                std = df_clean[col].std()
                if std > 0:
                    df_clean[col] = (df_clean[col] - mean) / std
    
    return df_clean

def validate_dataframe(df: pd.DataFrame, 
                       required_columns: List[str],
                       min_rows: int = 1) -> bool:
    """
    Validate that a DataFrame meets basic requirements.
    """
    if len(df) < min_rows:
        return False
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        return False
    
    return True

def export_clean_data(df: pd.DataFrame, 
                      output_path: str,
                      format: str = 'csv') -> None:
    """
    Export cleaned DataFrame to specified format.
    """
    if format == 'csv':
        df.to_csv(output_path, index=False)
    elif format == 'excel':
        df.to_excel(output_path, index=False)
    elif format == 'json':
        df.to_json(output_path, orient='records')
    else:
        raise ValueError(f"Unsupported format: {format}")
import pandas as pd

def clean_dataframe(df, column_name, condition_func):
    """
    Filters a pandas DataFrame based on a condition applied to a specific column.
    Removes rows where the condition function returns False.

    Args:
        df (pd.DataFrame): The input DataFrame to clean.
        column_name (str): The name of the column to apply the condition to.
        condition_func (function): A function that takes a single value from the
                                   specified column and returns a boolean.

    Returns:
        pd.DataFrame: A new DataFrame with rows filtered based on the condition.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")

    filtered_df = df[df[column_name].apply(condition_func)].copy()
    filtered_df.reset_index(drop=True, inplace=True)
    return filtered_df

def remove_outliers_iqr(df, column_name):
    """
    Removes outliers from a specific column in a DataFrame using the IQR method.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the numeric column to process.

    Returns:
        pd.DataFrame: A new DataFrame with outliers removed from the specified column.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")

    if not pd.api.types.is_numeric_dtype(df[column_name]):
        raise TypeError(f"Column '{column_name}' must be numeric for IQR outlier removal.")

    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    filtered_df = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)].copy()
    filtered_df.reset_index(drop=True, inplace=True)
    return filtered_df