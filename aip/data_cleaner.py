
import pandas as pd
import numpy as np
from pathlib import Path

def clean_csv_data(input_path, output_path=None):
    """
    Clean CSV data by handling missing values and removing duplicates.
    """
    try:
        df = pd.read_csv(input_path)
        
        original_rows = len(df)
        original_columns = len(df.columns)
        
        print(f"Original data: {original_rows} rows, {original_columns} columns")
        
        df_cleaned = df.copy()
        
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        categorical_cols = df_cleaned.select_dtypes(exclude=[np.number]).columns
        
        for col in numeric_cols:
            if df_cleaned[col].isnull().sum() > 0:
                df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
        
        for col in categorical_cols:
            if df_cleaned[col].isnull().sum() > 0:
                df_cleaned[col].fillna(df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else 'Unknown', inplace=True)
        
        df_cleaned.drop_duplicates(inplace=True)
        
        cleaned_rows = len(df_cleaned)
        rows_removed = original_rows - cleaned_rows
        
        print(f"Cleaned data: {cleaned_rows} rows, {original_columns} columns")
        print(f"Rows removed: {rows_removed}")
        
        if output_path is None:
            input_file = Path(input_path)
            output_path = input_file.parent / f"{input_file.stem}_cleaned{input_file.suffix}"
        
        df_cleaned.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
        
        return df_cleaned, output_path
        
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        return None, None
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty")
        return None, None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None, None

def validate_dataframe(df):
    """
    Validate dataframe for common data quality issues.
    """
    if df is None or df.empty:
        print("DataFrame is empty or None")
        return False
    
    issues = []
    
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        issues.append(f"Found {missing_values} missing values")
    
    duplicate_rows = df.duplicated().sum()
    if duplicate_rows > 0:
        issues.append(f"Found {duplicate_rows} duplicate rows")
    
    zero_variance_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if zero_variance_cols:
        issues.append(f"Columns with zero variance: {zero_variance_cols}")
    
    if issues:
        print("Data quality issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    print("Data validation passed")
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 6],
        'name': ['Alice', 'Bob', 'Charlie', None, 'Eve', 'Alice'],
        'age': [25, 30, None, 35, 40, 25],
        'score': [85.5, 92.0, 78.5, None, 88.0, 85.5],
        'department': ['HR', 'IT', 'IT', 'Finance', None, 'HR']
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned_df, output_file = clean_csv_data('test_data.csv')
    
    if cleaned_df is not None:
        validation_result = validate_dataframe(cleaned_df)
        
        print("\nSample of cleaned data:")
        print(cleaned_df.head())
        
        import os
        if os.path.exists('test_data.csv'):
            os.remove('test_data.csv')
        if os.path.exists(output_file):
            os.remove(output_file)
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        multiplier: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data.copy()

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        threshold: Z-score threshold (default 3)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    mask = z_scores < threshold
    filtered_data = data[mask]
    return filtered_data.copy()

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return pd.Series([0.5] * len(data), index=data.index)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with standardized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(data, numeric_columns=None, outlier_method='iqr', normalize_method='minmax'):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric columns to process (default: all numeric columns)
        outlier_method: 'iqr' or 'zscore' (default: 'iqr')
        normalize_method: 'minmax' or 'zscore' (default: 'minmax')
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column not in cleaned_data.columns:
            continue
            
        if outlier_method == 'iqr':
            cleaned_data = remove_outliers_iqr(cleaned_data, column)
        elif outlier_method == 'zscore':
            cleaned_data = remove_outliers_zscore(cleaned_data, column)
        else:
            raise ValueError(f"Unknown outlier method: {outlier_method}")
    
    for column in numeric_columns:
        if column not in cleaned_data.columns:
            continue
            
        if normalize_method == 'minmax':
            cleaned_data[f"{column}_normalized"] = normalize_minmax(cleaned_data, column)
        elif normalize_method == 'zscore':
            cleaned_data[f"{column}_standardized"] = normalize_zscore(cleaned_data, column)
        else:
            raise ValueError(f"Unknown normalize method: {normalize_method}")
    
    return cleaned_data

def get_cleaning_stats(original_data, cleaned_data, numeric_columns=None):
    """
    Generate statistics about the cleaning process.
    
    Args:
        original_data: original DataFrame
        cleaned_data: cleaned DataFrame
        numeric_columns: list of numeric columns to analyze
    
    Returns:
        Dictionary with cleaning statistics
    """
    if numeric_columns is None:
        numeric_columns = original_data.select_dtypes(include=[np.number]).columns.tolist()
    
    stats_dict = {
        'original_rows': len(original_data),
        'cleaned_rows': len(cleaned_data),
        'rows_removed': len(original_data) - len(cleaned_data),
        'removal_percentage': (len(original_data) - len(cleaned_data)) / len(original_data) * 100,
        'columns_processed': numeric_columns,
        'column_stats': {}
    }
    
    for column in numeric_columns:
        if column in original_data.columns and column in cleaned_data.columns:
            original_stats = {
                'mean': original_data[column].mean(),
                'std': original_data[column].std(),
                'min': original_data[column].min(),
                'max': original_data[column].max()
            }
            
            cleaned_stats = {
                'mean': cleaned_data[column].mean(),
                'std': cleaned_data[column].std(),
                'min': cleaned_data[column].min(),
                'max': cleaned_data[column].max()
            }
            
            stats_dict['column_stats'][column] = {
                'original': original_stats,
                'cleaned': cleaned_stats
            }
    
    return stats_dict