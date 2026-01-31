
import pandas as pd

def clean_dataframe(df, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
        fill_missing (str or dict): Method to fill missing values. 
            Options: 'mean', 'median', 'mode', or a dictionary of column:value pairs.
            If None, missing values are not filled.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
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
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame for required columns and basic integrity.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"import pandas as pd
import numpy as np

def clean_csv_data(file_path, output_path=None):
    """
    Load a CSV file, perform basic cleaning operations,
    and optionally save the cleaned data.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

    original_shape = df.shape
    print(f"Original data shape: {original_shape}")

    # Remove duplicate rows
    df.drop_duplicates(inplace=True)

    # Fill missing numeric values with column median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)

    # Fill missing categorical values with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
            df[col].fillna(mode_val, inplace=True)

    # Remove columns with more than 50% missing values
    threshold = len(df) * 0.5
    cols_to_drop = [col for col in df.columns if df[col].isnull().sum() > threshold]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
        print(f"Dropped columns with >50% missing values: {cols_to_drop}")

    cleaned_shape = df.shape
    print(f"Cleaned data shape: {cleaned_shape}")
    print(f"Removed {original_shape[0] - cleaned_shape[0]} duplicate rows.")

    if output_path:
        try:
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
        except Exception as e:
            print(f"Error saving file: {e}")

    return df

def summarize_data(df):
    """
    Generate basic summary statistics for the dataframe.
    """
    if df is None or df.empty:
        print("DataFrame is empty or None.")
        return

    print("\n=== DATA SUMMARY ===")
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print("\nColumn data types:")
    print(df.dtypes.value_counts())

    print("\nMissing values per column:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("No missing values.")
    else:
        for col, count in missing.items():
            percentage = (count / len(df)) * 100
            print(f"{col}: {count} ({percentage:.2f}%)")

if __name__ == "__main__":
    # Example usage
    input_file = "sample_data.csv"
    output_file = "cleaned_data.csv"

    cleaned_df = clean_csv_data(input_file, output_file)
    if cleaned_df is not None:
        summarize_data(cleaned_df)