import pandas as pd
import numpy as np

def load_data(filepath):
    """Load dataset from CSV file."""
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return None

def remove_outliers_iqr(df, column):
    """Remove outliers using IQR method for specified column."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    removed_count = len(df) - len(filtered_df)
    print(f"Removed {removed_count} outliers from column '{column}'")
    return filtered_df

def normalize_column(df, column):
    """Normalize column values to range [0, 1]."""
    if column in df.columns:
        min_val = df[column].min()
        max_val = df[column].max()
        
        if max_val != min_val:
            df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
            print(f"Column '{column}' normalized successfully.")
        else:
            print(f"Warning: Column '{column}' has constant values.")
    else:
        print(f"Error: Column '{column}' not found in dataframe.")
    return df

def clean_dataset(filepath, numeric_columns):
    """Main function to clean dataset."""
    df = load_data(filepath)
    
    if df is None:
        return None
    
    print("\nInitial data info:")
    print(df.info())
    
    original_shape = df.shape
    
    for column in numeric_columns:
        if column in df.columns:
            df = remove_outliers_iqr(df, column)
    
    for column in numeric_columns:
        if column in df.columns:
            df = normalize_column(df, column)
    
    print(f"\nCleaning complete.")
    print(f"Original shape: {original_shape}")
    print(f"Final shape: {df.shape}")
    
    return df

if __name__ == "__main__":
    data_file = "sample_data.csv"
    numeric_cols = ['age', 'income', 'score']
    
    cleaned_data = clean_dataset(data_file, numeric_cols)
    
    if cleaned_data is not None:
        output_file = "cleaned_data.csv"
        cleaned_data.to_csv(output_file, index=False)
        print(f"\nCleaned data saved to '{output_file}'")import pandas as pd
import numpy as np

def clean_missing_data(df, strategy='mean', columns=None):
    """
    Clean missing values in a DataFrame using specified strategy.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        strategy (str): Strategy for handling missing values. 
                       Options: 'mean', 'median', 'mode', 'drop', 'fill_zero'
        columns (list): List of columns to apply cleaning. If None, applies to all numeric columns.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if columns is None:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        if strategy == 'drop':
            df_clean = df_clean.dropna(subset=[col])
        elif strategy == 'mean':
            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
        elif strategy == 'median':
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        elif strategy == 'mode':
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
        elif strategy == 'fill_zero':
            df_clean[col] = df_clean[col].fillna(0)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
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

def load_and_clean_csv(filepath, **kwargs):
    """
    Load CSV file and clean missing data.
    
    Args:
        filepath (str): Path to CSV file
        **kwargs: Additional arguments passed to clean_missing_data
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    try:
        df = pd.read_csv(filepath)
        is_valid, message = validate_dataframe(df)
        
        if not is_valid:
            raise ValueError(f"Data validation failed: {message}")
        
        return clean_missing_data(df, **kwargs)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except pd.errors.EmptyDataError:
        raise ValueError("CSV file is empty")
    except Exception as e:
        raise RuntimeError(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, 2, 3, np.nan, 5],
        'C': [1, 2, 3, 4, 5]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_missing_data(df, strategy='mean')
    print("\nCleaned DataFrame (mean imputation):")
    print(cleaned)