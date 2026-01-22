import pandas as pd

def clean_dataset(df):
    """
    Clean a pandas DataFrame by removing duplicate rows and
    filling missing values with column means for numeric columns.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()
    
    # Fill missing numeric values with column mean
    numeric_cols = df_cleaned.select_dtypes(include=['number']).columns
    df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].mean())
    
    # For non-numeric columns, fill with mode (most frequent value)
    non_numeric_cols = df_cleaned.select_dtypes(exclude=['number']).columns
    for col in non_numeric_cols:
        if df_cleaned[col].isnull().any():
            mode_value = df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else 'Unknown'
            df_cleaned[col] = df_cleaned[col].fillna(mode_value)
    
    return df_cleaned

def validate_data(df, required_columns):
    """
    Validate that the DataFrame contains all required columns
    and has no completely empty columns.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for completely empty columns
    empty_columns = df.columns[df.isnull().all()].tolist()
    
    if empty_columns:
        print(f"Warning: Found completely empty columns: {empty_columns}")
    
    return True