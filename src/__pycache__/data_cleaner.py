import pandas as pd
import numpy as np

def clean_csv_data(filepath, drop_na=True, fill_strategy='mean'):
    """
    Load and clean a CSV file.
    
    Args:
        filepath (str): Path to the CSV file.
        drop_na (bool): If True, drop rows with missing values.
        fill_strategy (str): Strategy to fill missing values if drop_na is False.
                             Options: 'mean', 'median', 'mode', or 'zero'.
    
    Returns:
        pandas.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    
    original_shape = df.shape
    
    if drop_na:
        df_cleaned = df.dropna()
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
                    fill_value = df_cleaned[col].mean()
                
                df_cleaned[col].fillna(fill_value, inplace=True)
    
    cleaned_shape = df_cleaned.shape
    rows_removed = original_shape[0] - cleaned_shape[0]
    
    print(f"Original data shape: {original_shape}")
    print(f"Cleaned data shape: {cleaned_shape}")
    print(f"Rows removed: {rows_removed}")
    
    return df_cleaned

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df (pandas.DataFrame): Input DataFrame.
        subset (list): Columns to consider for identifying duplicates.
        keep (str): Which duplicates to keep. Options: 'first', 'last', False.
    
    Returns:
        pandas.DataFrame: DataFrame with duplicates removed.
    """
    if df is None or df.empty:
        return df
    
    original_count = len(df)
    df_deduped = df.drop_duplicates(subset=subset, keep=keep)
    removed_count = original_count - len(df_deduped)
    
    print(f"Original row count: {original_count}")
    print(f"Deduplicated row count: {len(df_deduped)}")
    print(f"Duplicates removed: {removed_count}")
    
    return df_deduped

def normalize_numeric_columns(df, columns=None, method='minmax'):
    """
    Normalize numeric columns in DataFrame.
    
    Args:
        df (pandas.DataFrame): Input DataFrame.
        columns (list): Columns to normalize. If None, normalize all numeric columns.
        method (str): Normalization method. Options: 'minmax', 'zscore'.
    
    Returns:
        pandas.DataFrame: DataFrame with normalized columns.
    """
    if df is None or df.empty:
        return df
    
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        numeric_cols = [col for col in columns if col in df.columns and np.issubdtype(df[col].dtype, np.number)]
    
    df_normalized = df.copy()
    
    for col in numeric_cols:
        if method == 'minmax':
            col_min = df_normalized[col].min()
            col_max = df_normalized[col].max()
            if col_max != col_min:
                df_normalized[col] = (df_normalized[col] - col_min) / (col_max - col_min)
        elif method == 'zscore':
            col_mean = df_normalized[col].mean()
            col_std = df_normalized[col].std()
            if col_std != 0:
                df_normalized[col] = (df_normalized[col] - col_mean) / col_std
    
    print(f"Normalized columns: {numeric_cols}")
    print(f"Normalization method: {method}")
    
    return df_normalized

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 6, 7, 8],
        'value': [10.5, 20.3, None, 15.7, 20.3, 12.1, None, 18.9],
        'category': ['A', 'B', 'A', 'C', 'B', 'A', 'B', 'C']
    }
    
    df = pd.DataFrame(sample_data)
    print("Sample DataFrame:")
    print(df)
    print("\n")
    
    cleaned_df = clean_csv_data('dummy_path', drop_na=False, fill_strategy='mean')
    
    test_df = pd.DataFrame({
        'A': [1, 1, 2, 3, 3],
        'B': ['x', 'x', 'y', 'z', 'z']
    })
    
    deduped = remove_duplicates(test_df, subset=['A', 'B'])
    print("\nDeduplication test:")
    print(deduped)
    
    numeric_df = pd.DataFrame({
        'score1': [10, 20, 30, 40, 50],
        'score2': [100, 200, 300, 400, 500]
    })
    
    normalized = normalize_numeric_columns(numeric_df, method='minmax')
    print("\nNormalization test:")
    print(normalized)