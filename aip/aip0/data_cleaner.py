
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, columns, factor=1.5):
    """
    Remove outliers using Interquartile Range method
    """
    df_clean = dataframe.copy()
    for col in columns:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def remove_outliers_zscore(dataframe, columns, threshold=3):
    """
    Remove outliers using Z-score method
    """
    df_clean = dataframe.copy()
    for col in columns:
        if col in df_clean.columns:
            z_scores = np.abs(stats.zscore(df_clean[col]))
            df_clean = df_clean[z_scores < threshold]
    return df_clean

def normalize_minmax(dataframe, columns):
    """
    Normalize data using Min-Max scaling
    """
    df_normalized = dataframe.copy()
    for col in columns:
        if col in df_normalized.columns:
            min_val = df_normalized[col].min()
            max_val = df_normalized[col].max()
            if max_val != min_val:
                df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
    return df_normalized

def normalize_zscore(dataframe, columns):
    """
    Normalize data using Z-score standardization
    """
    df_normalized = dataframe.copy()
    for col in columns:
        if col in df_normalized.columns:
            mean_val = df_normalized[col].mean()
            std_val = df_normalized[col].std()
            if std_val != 0:
                df_normalized[col] = (df_normalized[col] - mean_val) / std_val
    return df_normalized

def handle_missing_values(dataframe, strategy='mean', columns=None):
    """
    Handle missing values with different strategies
    """
    df_filled = dataframe.copy()
    if columns is None:
        columns = df_filled.columns
    
    for col in columns:
        if col in df_filled.columns and df_filled[col].isnull().any():
            if strategy == 'mean':
                fill_value = df_filled[col].mean()
            elif strategy == 'median':
                fill_value = df_filled[col].median()
            elif strategy == 'mode':
                fill_value = df_filled[col].mode()[0]
            elif strategy == 'drop':
                df_filled = df_filled.dropna(subset=[col])
                continue
            else:
                fill_value = 0
            
            df_filled[col] = df_filled[col].fillna(fill_value)
    
    return df_filled

def clean_data_pipeline(dataframe, config):
    """
    Execute complete data cleaning pipeline based on configuration
    """
    df_clean = dataframe.copy()
    
    if 'missing_values' in config:
        df_clean = handle_missing_values(
            df_clean,
            strategy=config['missing_values'].get('strategy', 'mean'),
            columns=config['missing_values'].get('columns')
        )
    
    if 'outliers' in config:
        method = config['outliers'].get('method', 'iqr')
        columns = config['outliers'].get('columns', df_clean.columns.tolist())
        
        if method == 'iqr':
            factor = config['outliers'].get('factor', 1.5)
            df_clean = remove_outliers_iqr(df_clean, columns, factor)
        elif method == 'zscore':
            threshold = config['outliers'].get('threshold', 3)
            df_clean = remove_outliers_zscore(df_clean, columns, threshold)
    
    if 'normalization' in config:
        method = config['normalization'].get('method', 'minmax')
        columns = config['normalization'].get('columns', df_clean.columns.tolist())
        
        if method == 'minmax':
            df_clean = normalize_minmax(df_clean, columns)
        elif method == 'zscore':
            df_clean = normalize_zscore(df_clean, columns)
    
    return df_clean
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            df = remove_outliers_iqr(df, col)
            df = normalize_minmax(df, col)
        
        cleaned_file = file_path.replace('.csv', '_cleaned.csv')
        df.to_csv(cleaned_file, index=False)
        return f"Cleaned data saved to {cleaned_file}"
    
    except FileNotFoundError:
        return "Error: File not found"
    except Exception as e:
        return f"Error during processing: {str(e)}"

if __name__ == "__main__":
    result = clean_dataset('sample_data.csv')
    print(result)
import pandas as pd
import numpy as np

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

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

def clean_dataset(df, columns_to_clean=None):
    """
    Clean multiple columns in a DataFrame by removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns_to_clean (list): List of column names to clean. If None, clean all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    dict: Dictionary of statistics for each cleaned column
    """
    if columns_to_clean is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns_to_clean = list(numeric_cols)
    
    cleaned_df = df.copy()
    statistics = {}
    
    for column in columns_to_clean:
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            
            stats = calculate_summary_statistics(cleaned_df, column)
            stats['outliers_removed'] = removed_count
            statistics[column] = stats
    
    return cleaned_df, statistics

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[::100, 'A'] = 500
    
    print("Original DataFrame shape:", df.shape)
    print("\nOriginal summary statistics:")
    print(df.describe())
    
    cleaned_df, stats = clean_dataset(df, ['A', 'B'])
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("\nCleaning statistics:")
    for col, col_stats in stats.items():
        print(f"\n{col}:")
        for stat_name, stat_value in col_stats.items():
            print(f"  {stat_name}: {stat_value:.2f}")