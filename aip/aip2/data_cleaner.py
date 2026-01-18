import pandas as pd
import numpy as np
import sys

def clean_csv(input_file, output_file):
    try:
        df = pd.read_csv(input_file)
        
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        df[categorical_cols] = df[categorical_cols].fillna('Unknown')
        
        df.drop_duplicates(inplace=True)
        
        df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to {output_file}")
        return True
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return False
    except pd.errors.EmptyDataError:
        print(f"Error: File '{input_file}' is empty.")
        return False
    except Exception as e:
        print(f"Error during cleaning: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python data_cleaner.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    success = clean_csv(input_file, output_file)
    sys.exit(0 if success else 1)
import pandas as pd
import numpy as np

def clean_missing_data(df, strategy='mean', columns=None):
    """
    Handle missing values in a DataFrame using specified strategy.
    
    Args:
        df: pandas DataFrame
        strategy: Method for handling missing values ('mean', 'median', 'mode', 'drop')
        columns: List of columns to clean, if None cleans all columns
    
    Returns:
        Cleaned DataFrame
    """
    if df.empty:
        return df
    
    if columns is None:
        columns = df.columns
    
    df_clean = df.copy()
    
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        if df_clean[col].isnull().sum() == 0:
            continue
            
        if strategy == 'drop':
            df_clean = df_clean.dropna(subset=[col])
        elif strategy == 'mean':
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
        elif strategy == 'median':
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
        elif strategy == 'mode':
            if not df_clean[col].empty:
                mode_value = df_clean[col].mode()
                if not mode_value.empty:
                    df_clean[col].fillna(mode_value[0], inplace=True)
    
    return df_clean

def remove_outliers(df, columns=None, threshold=3):
    """
    Remove outliers using z-score method.
    
    Args:
        df: pandas DataFrame
        columns: List of columns to check for outliers
        threshold: Z-score threshold for outlier detection
    
    Returns:
        DataFrame with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        if pd.api.types.is_numeric_dtype(df_clean[col]):
            z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
            df_clean = df_clean[z_scores < threshold]
    
    return df_clean

def normalize_data(df, columns=None):
    """
    Normalize numeric columns to range [0, 1].
    
    Args:
        df: pandas DataFrame
        columns: List of columns to normalize
    
    Returns:
        Normalized DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_normalized = df.copy()
    
    for col in columns:
        if col not in df_normalized.columns:
            continue
            
        if pd.api.types.is_numeric_dtype(df_normalized[col]):
            col_min = df_normalized[col].min()
            col_max = df_normalized[col].max()
            
            if col_max != col_min:
                df_normalized[col] = (df_normalized[col] - col_min) / (col_max - col_min)
    
    return df_normalized

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: List of columns that must be present
    
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