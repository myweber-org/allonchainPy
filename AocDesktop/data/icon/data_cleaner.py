import pandas as pd
import numpy as np

def remove_missing_rows(df, columns=None):
    """
    Remove rows with missing values from DataFrame.
    
    Args:
        df: pandas DataFrame
        columns: list of columns to check for missing values.
                 If None, checks all columns.
    
    Returns:
        Cleaned DataFrame
    """
    if columns is None:
        columns = df.columns
    
    return df.dropna(subset=columns)

def fill_missing_with_mean(df, columns=None):
    """
    Fill missing values with column mean.
    
    Args:
        df: pandas DataFrame
        columns: list of columns to fill.
                 If None, fills all numeric columns.
    
    Returns:
        DataFrame with filled values
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_filled = df.copy()
    for col in columns:
        if col in df.columns and df[col].dtype in [np.float64, np.int64]:
            df_filled[col] = df[col].fillna(df[col].mean())
    
    return df_filled

def detect_outliers_iqr(df, column, multiplier=1.5):
    """
    Detect outliers using IQR method.
    
    Args:
        df: pandas DataFrame
        column: column name to check for outliers
        multiplier: IQR multiplier (default: 1.5)
    
    Returns:
        Boolean Series indicating outliers
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return (df[column] < lower_bound) | (df[column] > upper_bound)

def remove_outliers(df, column, multiplier=1.5):
    """
    Remove outliers from DataFrame.
    
    Args:
        df: pandas DataFrame
        column: column name to check for outliers
        multiplier: IQR multiplier (default: 1.5)
    
    Returns:
        DataFrame without outliers
    """
    outliers = detect_outliers_iqr(df, column, multiplier)
    return df[~outliers]

def normalize_column(df, column, method='minmax'):
    """
    Normalize column values.
    
    Args:
        df: pandas DataFrame
        column: column name to normalize
        method: normalization method ('minmax' or 'zscore')
    
    Returns:
        DataFrame with normalized column
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_normalized = df.copy()
    
    if method == 'minmax':
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val != min_val:
            df_normalized[column] = (df[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = df[column].mean()
        std_val = df[column].std()
        if std_val != 0:
            df_normalized[column] = (df[column] - mean_val) / std_val
    
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")
    
    return df_normalized

def clean_dataset(df, numeric_columns=None, outlier_multiplier=1.5):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: pandas DataFrame
        numeric_columns: list of numeric columns to process
        outlier_multiplier: IQR multiplier for outlier detection
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove rows with missing values in numeric columns
    df_clean = remove_missing_rows(df, numeric_columns)
    
    # Fill remaining missing values with mean
    df_clean = fill_missing_with_mean(df_clean, numeric_columns)
    
    # Remove outliers from each numeric column
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean = remove_outliers(df_clean, col, outlier_multiplier)
    
    # Normalize numeric columns
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean = normalize_column(df_clean, col, method='minmax')
    
    return df_clean.reset_index(drop=True)import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, threshold=1.5):
    """
    Remove outliers using IQR method.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data.copy()

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    filtered_indices = np.where(z_scores < threshold)[0]
    
    filtered_data = data.iloc[filtered_indices].copy()
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].copy()
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].copy()
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(df, numeric_columns=None, outlier_method='iqr', normalize_method='minmax'):
    """
    Main function to clean dataset by removing outliers and normalizing numeric columns.
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col not in cleaned_df.columns:
            continue
            
        if outlier_method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        elif outlier_method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, col)
        
        if normalize_method == 'minmax':
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
        elif normalize_method == 'zscore':
            cleaned_df[col] = normalize_zscore(cleaned_df, col)
    
    return cleaned_df.reset_index(drop=True)

def validate_data(df, required_columns=None, check_missing=True, check_duplicates=True):
    """
    Validate dataset for common data quality issues.
    """
    validation_report = {}
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_report['missing_columns'] = missing_columns
    
    if check_missing:
        missing_values = df.isnull().sum().to_dict()
        validation_report['missing_values'] = {k: v for k, v in missing_values.items() if v > 0}
    
    if check_duplicates:
        duplicate_count = df.duplicated().sum()
        validation_report['duplicate_rows'] = duplicate_count
    
    return validation_report
import pandas as pd
import numpy as np
from pathlib import Path

def load_and_clean_csv(file_path, fill_strategy='mean', drop_threshold=0.5):
    """
    Load a CSV file and perform basic cleaning operations.
    
    Args:
        file_path (str): Path to the CSV file
        fill_strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', 'zero')
        drop_threshold (float): Drop columns with missing values above this threshold (0.0 to 1.0)
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        
        # Drop columns with too many missing values
        missing_ratio = df.isnull().sum() / len(df)
        columns_to_drop = missing_ratio[missing_ratio > drop_threshold].index
        if len(columns_to_drop) > 0:
            df = df.drop(columns=columns_to_drop)
            print(f"Dropped {len(columns_to_drop)} columns with >{drop_threshold*100}% missing values")
        
        # Fill missing values based on strategy
        if fill_strategy == 'mean':
            df = df.fillna(df.select_dtypes(include=[np.number]).mean())
        elif fill_strategy == 'median':
            df = df.fillna(df.select_dtypes(include=[np.number]).median())
        elif fill_strategy == 'mode':
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        elif fill_strategy == 'zero':
            df = df.fillna(0)
        
        # Remove duplicate rows
        initial_rows = len(df)
        df = df.drop_duplicates()
        if len(df) < initial_rows:
            print(f"Removed {initial_rows - len(df)} duplicate rows")
        
        # Reset index
        df = df.reset_index(drop=True)
        
        print(f"Cleaned data: {len(df)} rows and {len(df.columns)} columns remaining")
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error during cleaning: {str(e)}")
        return None

def save_cleaned_data(df, output_path):
    """
    Save cleaned DataFrame to CSV.
    
    Args:
        df (pd.DataFrame): Cleaned DataFrame
        output_path (str): Path to save the cleaned data
    """
    if df is not None:
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to {output_path}")
            return True
        except Exception as e:
            print(f"Error saving file: {str(e)}")
            return False
    return False

def get_data_summary(df):
    """
    Generate summary statistics for the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
    
    Returns:
        dict: Summary statistics
    """
    if df is None or df.empty:
        return {}
    
    summary = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': len(df.select_dtypes(include=['object']).columns),
        'missing_values': int(df.isnull().sum().sum()),
        'duplicate_rows': len(df) - len(df.drop_duplicates()),
        'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
    }
    
    return summary

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = load_and_clean_csv(input_file, fill_strategy='mean', drop_threshold=0.3)
    
    if cleaned_df is not None:
        summary = get_data_summary(cleaned_df)
        print("Data Summary:", summary)
        
        save_cleaned_data(cleaned_df, output_file)
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Args:
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

def calculate_summary_stats(df, column):
    """
    Calculate summary statistics for a column.
    
    Args:
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
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

def clean_dataset(df, numeric_columns=None):
    """
    Clean dataset by removing outliers from all numeric columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        numeric_columns (list): List of numeric column names. If None, uses all numeric columns.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'id': range(1, 101),
        'value': np.random.normal(100, 15, 100)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[10, 'value'] = 500
    df.loc[20, 'value'] = -200
    
    print("Original dataset shape:", df.shape)
    print("Summary statistics before cleaning:")
    print(calculate_summary_stats(df, 'value'))
    
    cleaned_df = clean_dataset(df, ['value'])
    
    print("\nCleaned dataset shape:", cleaned_df.shape)
    print("Summary statistics after cleaning:")
    print(calculate_summary_stats(cleaned_df, 'value'))