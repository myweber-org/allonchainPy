
import numpy as np

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data
import pandas as pd
import numpy as np
from typing import Optional

def clean_csv_data(file_path: str, 
                   missing_strategy: str = 'drop',
                   fill_value: Optional[float] = None) -> pd.DataFrame:
    """
    Load and clean CSV data by handling missing values.
    
    Args:
        file_path: Path to CSV file
        missing_strategy: Strategy for handling missing values ('drop', 'fill', 'interpolate')
        fill_value: Value to fill missing data with when using 'fill' strategy
    
    Returns:
        Cleaned pandas DataFrame
    """
    try:
        df = pd.read_csv(file_path)
        
        if missing_strategy == 'drop':
            df_cleaned = df.dropna()
        elif missing_strategy == 'fill':
            if fill_value is not None:
                df_cleaned = df.fillna(fill_value)
            else:
                df_cleaned = df.fillna(df.mean(numeric_only=True))
        elif missing_strategy == 'interpolate':
            df_cleaned = df.interpolate(method='linear', limit_direction='forward')
        else:
            raise ValueError(f"Unknown missing strategy: {missing_strategy}")
        
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        if not numeric_cols.empty:
            df_cleaned[numeric_cols] = df_cleaned[numeric_cols].round(3)
        
        return df_cleaned
    
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        raise
    except pd.errors.EmptyDataError:
        print("Error: CSV file is empty")
        raise
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        raise

def validate_dataframe(df: pd.DataFrame, 
                      required_columns: list = None) -> bool:
    """
    Validate DataFrame structure and content.
    
    Args:
        df: DataFrame to validate
        required_columns: List of column names that must be present
    
    Returns:
        Boolean indicating if DataFrame is valid
    """
    if df.empty:
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return False
    
    if df.isnull().any().any():
        print("Warning: DataFrame contains missing values")
    
    return True

def save_cleaned_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save cleaned DataFrame to CSV file.
    
    Args:
        df: Cleaned DataFrame
        output_path: Path to save the cleaned data
    """
    try:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")
    except Exception as e:
        print(f"Error saving cleaned data: {str(e)}")
        raise

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [5.123, np.nan, 7.456, 8.789, 9.012],
        'C': ['x', 'y', 'z', np.nan, 'w']
    })
    
    sample_data.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_csv_data('sample_data.csv', missing_strategy='fill')
    
    if validate_dataframe(cleaned_df, required_columns=['A', 'B', 'C']):
        save_cleaned_data(cleaned_df, 'cleaned_sample_data.csv')
        print("Data cleaning completed successfully")
    else:
        print("Data validation failed")import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_missing (str): Method to fill missing values. 
                        Options: 'mean', 'median', 'mode', or 'drop'. Default is 'mean'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median', 'mode']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
            elif fill_missing == 'median':
                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
            elif fill_missing == 'mode':
                cleaned_df[col].fillna(cleaned_df[col].mode()[0], inplace=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame for basic integrity checks.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, None],
        'B': [5, None, 7, 8, 9],
        'C': ['x', 'y', 'x', 'z', 'w']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame (drop duplicates, fill with mean):")
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print(cleaned)
    
    is_valid, message = validate_dataframe(cleaned, required_columns=['A', 'B', 'C'])
    print(f"\nValidation: {is_valid}, Message: {message}")