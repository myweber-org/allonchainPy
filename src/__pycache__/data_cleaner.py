import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to remove duplicate rows. Default is True.
        fill_missing (str or dict): Method to fill missing values. Can be 'mean', 'median', 
                                   'mode', or a dictionary of column:value pairs. Default is None.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows.")
    
    if fill_missing is not None:
        if isinstance(fill_missing, dict):
            cleaned_df = cleaned_df.fillna(fill_missing)
        elif fill_missing == 'mean':
            cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
        elif fill_missing == 'median':
            cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
        elif fill_missing == 'mode':
            for col in cleaned_df.columns:
                if cleaned_df[col].dtype == 'object':
                    mode_val = cleaned_df[col].mode()
                    if not mode_val.empty:
                        cleaned_df[col] = cleaned_df[col].fillna(mode_val[0])
        else:
            raise ValueError("fill_missing must be 'mean', 'median', 'mode', or a dictionary.")
        
        print(f"Filled missing values using method: {fill_missing}")
    
    missing_count = cleaned_df.isnull().sum().sum()
    if missing_count > 0:
        print(f"Warning: {missing_count} missing values remain in the dataset.")
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the structure and content of a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
        min_rows (int): Minimum number of rows required.
    
    Returns:
        bool: True if validation passes, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame.")
        return False
    
    if len(df) < min_rows:
        print(f"Error: DataFrame has fewer than {min_rows} rows.")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'value': [10.5, None, 15.2, None, 20.1, 18.7],
        'category': ['A', 'B', 'B', 'C', None, 'A']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaning dataset...")
    
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid = validate_data(cleaned, required_columns=['id', 'value'], min_rows=3)
    print(f"\nData validation passed: {is_valid}")import pandas as pd
import numpy as np

def remove_duplicates(df):
    """Remove duplicate rows from DataFrame."""
    return df.drop_duplicates()

def fill_missing_values(df, strategy='mean'):
    """Fill missing values using specified strategy."""
    if strategy == 'mean':
        return df.fillna(df.mean(numeric_only=True))
    elif strategy == 'median':
        return df.fillna(df.median(numeric_only=True))
    elif strategy == 'mode':
        return df.fillna(df.mode().iloc[0])
    else:
        return df.fillna(0)

def normalize_column(df, column_name):
    """Normalize specified column to range [0,1]."""
    if column_name in df.columns:
        col_min = df[column_name].min()
        col_max = df[column_name].max()
        if col_max != col_min:
            df[column_name] = (df[column_name] - col_min) / (col_max - col_min)
    return df

def clean_dataframe(df, remove_dups=True, fill_strategy='mean', normalize_cols=None):
    """Apply multiple cleaning operations to DataFrame."""
    if remove_dups:
        df = remove_duplicates(df)
    
    df = fill_missing_values(df, strategy=fill_strategy)
    
    if normalize_cols:
        if isinstance(normalize_cols, str):
            normalize_cols = [normalize_cols]
        for col in normalize_cols:
            df = normalize_column(df, col)
    
    return df

def load_and_clean_csv(filepath, **kwargs):
    """Load CSV file and apply cleaning operations."""
    try:
        df = pd.read_csv(filepath)
        return clean_dataframe(df, **kwargs)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None
import pandas as pd
import numpy as np

def clean_csv_data(input_file, output_file):
    """
    Load a CSV file, handle missing values, and save cleaned data.
    """
    try:
        df = pd.read_csv(input_file)
        
        # Display initial info
        print(f"Original data shape: {df.shape}")
        print(f"Missing values per column:\n{df.isnull().sum()}")
        
        # Fill missing numeric values with column median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"Filled missing values in '{col}' with median: {median_val}")
        
        # Fill missing categorical values with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
                print(f"Filled missing values in '{col}' with mode: {mode_val}")
        
        # Remove duplicate rows
        initial_rows = len(df)
        df.drop_duplicates(inplace=True)
        duplicates_removed = initial_rows - len(df)
        print(f"Removed {duplicates_removed} duplicate rows")
        
        # Save cleaned data
        df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to: {output_file}")
        print(f"Final data shape: {df.shape}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage
    input_path = "raw_data.csv"
    output_path = "cleaned_data.csv"
    
    cleaned_df = clean_csv_data(input_path, output_path)
    
    if cleaned_df is not None:
        print("\nData cleaning completed successfully")
        print(cleaned_df.head())