
import pandas as pd
import numpy as np

def clean_csv_data(filepath, fill_strategy='mean', output_path=None):
    """
    Load a CSV file, handle missing values, and optionally save cleaned data.
    
    Args:
        filepath (str): Path to input CSV file.
        fill_strategy (str): Strategy for filling missing values.
            Options: 'mean', 'median', 'mode', 'zero', 'drop'.
        output_path (str, optional): Path to save cleaned CSV. If None, returns DataFrame.
    
    Returns:
        pandas.DataFrame or None: Cleaned DataFrame if output_path is None.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded data with shape: {df.shape}")
        
        missing_counts = df.isnull().sum()
        if missing_counts.any():
            print("Missing values per column:")
            print(missing_counts[missing_counts > 0])
            
            if fill_strategy == 'drop':
                df_cleaned = df.dropna()
                print(f"Removed rows with missing values. New shape: {df_cleaned.shape}")
            else:
                df_cleaned = df.copy()
                numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
                
                for col in numeric_cols:
                    if df_cleaned[col].isnull().any():
                        if fill_strategy == 'mean':
                            fill_value = df_cleaned[col].mean()
                        elif fill_strategy == 'median':
                            fill_value = df_cleaned[col].median()
                        elif fill_strategy == 'mode':
                            fill_value = df_cleaned[col].mode()[0]
                        elif fill_strategy == 'zero':
                            fill_value = 0
                        else:
                            raise ValueError(f"Unknown fill strategy: {fill_strategy}")
                        
                        df_cleaned[col].fillna(fill_value, inplace=True)
                        print(f"Filled missing values in '{col}' with {fill_strategy}: {fill_value:.4f}")
        else:
            df_cleaned = df
            print("No missing values found.")
        
        if output_path:
            df_cleaned.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
            return None
        else:
            return df_cleaned
            
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pandas.DataFrame): DataFrame to validate.
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
    
    print(f"Validation passed. Shape: {df.shape}, Columns: {list(df.columns)}")
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'temperature': [22.5, np.nan, 24.1, 23.8, np.nan, 25.2],
        'humidity': [45, 48, np.nan, 50, 52, 49],
        'pressure': [1013, 1012, 1014, np.nan, 1015, 1013]
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned = clean_csv_data('test_data.csv', fill_strategy='median', output_path='cleaned_data.csv')
    
    if cleaned is not None:
        is_valid = validate_dataframe(cleaned, required_columns=['temperature', 'humidity', 'pressure'])
        print(f"Data validation result: {is_valid}")