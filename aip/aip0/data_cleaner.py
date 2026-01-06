import pandas as pd
import numpy as np

def clean_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in a DataFrame using specified strategy.
    
    Args:
        df: pandas DataFrame containing data with potential missing values
        strategy: Method for imputation ('mean', 'median', 'mode', or 'drop')
        columns: List of column names to apply cleaning (None for all columns)
    
    Returns:
        Cleaned pandas DataFrame
    """
    if df.empty:
        return df
    
    if columns is None:
        columns = df.columns
    else:
        columns = [col for col in columns if col in df.columns]
    
    df_clean = df.copy()
    
    for column in columns:
        if df_clean[column].isnull().any():
            if strategy == 'mean':
                fill_value = df_clean[column].mean()
            elif strategy == 'median':
                fill_value = df_clean[column].median()
            elif strategy == 'mode':
                fill_value = df_clean[column].mode()[0] if not df_clean[column].mode().empty else np.nan
            elif strategy == 'drop':
                df_clean = df_clean.dropna(subset=[column])
                continue
            else:
                raise ValueError(f"Unsupported strategy: {strategy}")
            
            df_clean[column] = df_clean[column].fillna(fill_value)
    
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: List of column names that must be present
    
    Returns:
        Tuple of (is_valid, error_message)
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

def load_and_clean_csv(filepath, **kwargs):
    """
    Load CSV file and clean missing values.
    
    Args:
        filepath: Path to CSV file
        **kwargs: Additional arguments passed to clean_missing_values
    
    Returns:
        Cleaned pandas DataFrame
    """
    try:
        df = pd.read_csv(filepath)
        is_valid, message = validate_dataframe(df)
        
        if not is_valid:
            raise ValueError(f"Data validation failed: {message}")
        
        return clean_missing_values(df, **kwargs)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except pd.errors.EmptyDataError:
        raise ValueError("CSV file is empty")
    except Exception as e:
        raise RuntimeError(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, 2, 3, np.nan, 5],
        'C': [1, 2, 3, 4, 5]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_missing_values(df, strategy='mean')
    print("\nCleaned DataFrame (mean imputation):")
    print(cleaned_df)