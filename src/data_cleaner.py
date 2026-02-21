import pandas as pd
import numpy as np

def clean_csv_data(filepath, fill_method='mean', output_path=None):
    """
    Load a CSV file, handle missing values, and optionally save cleaned data.
    
    Args:
        filepath (str): Path to the input CSV file.
        fill_method (str): Method for filling missing values ('mean', 'median', 'mode', or 'zero').
        output_path (str, optional): Path to save cleaned CSV. If None, returns DataFrame.
    
    Returns:
        pd.DataFrame or None: Cleaned DataFrame if output_path is None, else None.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded data with shape: {df.shape}")
        
        missing_counts = df.isnull().sum()
        if missing_counts.any():
            print("Missing values per column:")
            for col, count in missing_counts[missing_counts > 0].items():
                print(f"  {col}: {count}")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if fill_method == 'mean':
                fill_values = df[numeric_cols].mean()
            elif fill_method == 'median':
                fill_values = df[numeric_cols].median()
            elif fill_method == 'mode':
                fill_values = df[numeric_cols].mode().iloc[0]
            elif fill_method == 'zero':
                fill_values = 0
            else:
                raise ValueError(f"Unknown fill method: {fill_method}")
            
            df[numeric_cols] = df[numeric_cols].fillna(fill_values)
            print(f"Filled missing values using {fill_method} method.")
        else:
            print("No missing values found.")
        
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
            return None
        else:
            return df
            
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {e}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list, optional): List of required column names.
    
    Returns:
        bool: True if validation passes, False otherwise.
    """
    if df is None or df.empty:
        print("Validation failed: DataFrame is empty or None.")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Validation failed: Missing required columns: {missing_cols}")
            return False
    
    if df.isnull().any().any():
        print("Validation warning: DataFrame still contains missing values.")
    
    return True

if __name__ == "__main__":
    cleaned_df = clean_csv_data('sample_data.csv', fill_method='median')
    if cleaned_df is not None:
        is_valid = validate_dataframe(cleaned_df)
        if is_valid:
            print("Data cleaning and validation completed successfully.")
        else:
            print("Data validation failed.")