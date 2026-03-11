
import pandas as pd
import numpy as np

def clean_dataset(df, id_column=None, drop_duplicates=True, standardize_columns=True):
    """
    Clean a pandas DataFrame by removing duplicates and standardizing column names.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    id_column (str, optional): Column to use for duplicate identification
    drop_duplicates (bool): Whether to remove duplicate rows
    standardize_columns (bool): Whether to standardize column names
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if standardize_columns:
        df_clean.columns = df_clean.columns.str.strip().str.lower().str.replace(' ', '_')
    
    if drop_duplicates:
        if id_column and id_column in df_clean.columns:
            df_clean = df_clean.drop_duplicates(subset=[id_column])
        else:
            df_clean = df_clean.drop_duplicates()
    
    return df_clean

def validate_data(df, required_columns=None, numeric_columns=None):
    """
    Validate DataFrame structure and data types.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    numeric_columns (list): List of columns that should be numeric
    
    Returns:
    dict: Validation results with status and messages
    """
    validation_result = {
        'is_valid': True,
        'messages': [],
        'missing_columns': [],
        'non_numeric_columns': []
    }
    
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            validation_result['is_valid'] = False
            validation_result['missing_columns'] = missing
            validation_result['messages'].append(f"Missing required columns: {missing}")
    
    if numeric_columns:
        non_numeric = []
        for col in numeric_columns:
            if col in df.columns:
                if not np.issubdtype(df[col].dtype, np.number):
                    non_numeric.append(col)
        
        if non_numeric:
            validation_result['is_valid'] = False
            validation_result['non_numeric_columns'] = non_numeric
            validation_result['messages'].append(f"Non-numeric columns found: {non_numeric}")
    
    return validation_result

def handle_missing_values(df, strategy='drop', fill_value=None, columns=None):
    """
    Handle missing values in DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): 'drop', 'fill', or 'interpolate'
    fill_value: Value to use for filling (if strategy='fill')
    columns (list): Specific columns to process
    
    Returns:
    pd.DataFrame: DataFrame with handled missing values
    """
    df_processed = df.copy()
    
    if columns:
        cols_to_process = [col for col in columns if col in df_processed.columns]
    else:
        cols_to_process = df_processed.columns.tolist()
    
    if strategy == 'drop':
        df_processed = df_processed.dropna(subset=cols_to_process)
    elif strategy == 'fill':
        if fill_value is not None:
            df_processed[cols_to_process] = df_processed[cols_to_process].fillna(fill_value)
        else:
            for col in cols_to_process:
                if np.issubdtype(df_processed[col].dtype, np.number):
                    df_processed[col] = df_processed[col].fillna(df_processed[col].median())
                else:
                    df_processed[col] = df_processed[col].fillna(df_processed[col].mode().iloc[0] if not df_processed[col].mode().empty else '')
    elif strategy == 'interpolate':
        df_processed[cols_to_process] = df_processed[cols_to_process].interpolate(method='linear')
    
    return df_processed