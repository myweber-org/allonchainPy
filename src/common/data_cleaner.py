
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def clean_dataset(input_path, output_path):
    try:
        df = pd.read_csv(input_path)
        print(f"Original dataset shape: {df.shape}")
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            original_count = len(df)
            df = remove_outliers_iqr(df, col)
            removed_count = original_count - len(df)
            if removed_count > 0:
                print(f"Removed {removed_count} outliers from column: {col}")
        
        df.to_csv(output_path, index=False)
        print(f"Cleaned dataset saved to: {output_path}")
        print(f"Final dataset shape: {df.shape}")
        
        return True
    except Exception as e:
        print(f"Error during cleaning: {str(e)}")
        return False

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    success = clean_dataset(input_file, output_file)
    
    if success:
        print("Data cleaning completed successfully")
    else:
        print("Data cleaning failed")import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: pandas DataFrame
        subset: column label or sequence of labels to consider for duplicates
    
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')

def fill_missing_values(df, strategy='mean', columns=None):
    """
    Fill missing values in DataFrame.
    
    Args:
        df: pandas DataFrame
        strategy: 'mean', 'median', 'mode', or 'constant'
        columns: list of columns to fill, if None fills all columns
    
    Returns:
        DataFrame with missing values filled
    """
    if columns is None:
        columns = df.columns
    
    df_filled = df.copy()
    
    for col in columns:
        if df[col].dtype in ['int64', 'float64']:
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mode':
                fill_value = df[col].mode()[0]
            elif strategy == 'constant':
                fill_value = 0
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            df_filled[col] = df[col].fillna(fill_value)
        else:
            # For non-numeric columns, fill with mode or empty string
            if strategy == 'mode':
                fill_value = df[col].mode()[0] if not df[col].mode().empty else ''
            elif strategy == 'constant':
                fill_value = ''
            else:
                fill_value = df[col].mode()[0] if not df[col].mode().empty else ''
            
            df_filled[col] = df[col].fillna(fill_value)
    
    return df_filled

def normalize_columns(df, columns=None):
    """
    Normalize numeric columns to range [0, 1].
    
    Args:
        df: pandas DataFrame
        columns: list of columns to normalize, if None normalizes all numeric columns
    
    Returns:
        DataFrame with normalized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_normalized = df.copy()
    
    for col in columns:
        if df[col].dtype in ['int64', 'float64']:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_max != col_min:
                df_normalized[col] = (df[col] - col_min) / (col_max - col_min)
            else:
                df_normalized[col] = 0
    
    return df_normalized

def detect_outliers(df, columns=None, threshold=3):
    """
    Detect outliers using z-score method.
    
    Args:
        df: pandas DataFrame
        columns: list of columns to check, if None checks all numeric columns
        threshold: z-score threshold for outlier detection
    
    Returns:
        DataFrame with outlier flags
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_with_flags = df.copy()
    
    for col in columns:
        if df[col].dtype in ['int64', 'float64']:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            df_with_flags[f'{col}_outlier'] = z_scores > threshold
    
    return df_with_flags

def clean_dataset(df, remove_dups=True, fill_na=True, normalize=False, detect_out=False):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: pandas DataFrame
        remove_dups: whether to remove duplicates
        fill_na: whether to fill missing values
        normalize: whether to normalize numeric columns
        detect_out: whether to detect outliers
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if remove_dups:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if fill_na:
        cleaned_df = fill_missing_values(cleaned_df, strategy='mean')
    
    if normalize:
        cleaned_df = normalize_columns(cleaned_df)
    
    if detect_out:
        cleaned_df = detect_outliers(cleaned_df)
    
    return cleaned_df
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def calculate_statistics(df, column):
    """
    Calculate basic statistics for a column after outlier removal.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing statistical measures
    """
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

def clean_dataset(df, numeric_columns):
    """
    Clean dataset by removing outliers from multiple numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    dict: Statistics before and after cleaning for each column
    """
    original_stats = {}
    cleaned_stats = {}
    
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            original_stats[col] = calculate_statistics(df, col)
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_stats[col] = calculate_statistics(cleaned_df, col)
    
    return cleaned_df, {'original': original_stats, 'cleaned': cleaned_stats}

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    sample_data = {
        'id': range(100),
        'value': np.random.normal(100, 15, 100),
        'score': np.random.uniform(0, 100, 100)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[95, 'value'] = 500  # Add outlier
    df.loc[96, 'score'] = 150  # Add outlier
    
    print("Original dataset shape:", df.shape)
    print("\nOriginal statistics:")
    print(df[['value', 'score']].describe())
    
    cleaned_df, stats = clean_dataset(df, ['value', 'score'])
    
    print("\nCleaned dataset shape:", cleaned_df.shape)
    print("\nCleaned statistics:")
    print(cleaned_df[['value', 'score']].describe())