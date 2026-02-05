
import pandas as pd
import numpy as np
from datetime import datetime

def clean_csv_data(input_path, output_path):
    """
    Clean CSV data by handling missing values, converting data types,
    and removing duplicate rows.
    """
    try:
        df = pd.read_csv(input_path)
        
        print(f"Original shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        df_cleaned = df.copy()
        
        df_cleaned = df_cleaned.drop_duplicates()
        print(f"After removing duplicates: {df_cleaned.shape}")
        
        numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df_cleaned[col].isnull().sum() > 0:
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
        
        categorical_columns = df_cleaned.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if df_cleaned[col].isnull().sum() > 0:
                df_cleaned[col] = df_cleaned[col].fillna('Unknown')
        
        date_columns = []
        for col in df_cleaned.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='coerce')
                    date_columns.append(col)
                except:
                    pass
        
        for col in numeric_columns:
            if df_cleaned[col].dtype in ['int64', 'float64']:
                q1 = df_cleaned[col].quantile(0.25)
                q3 = df_cleaned[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = ((df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)).sum()
                if outliers > 0:
                    df_cleaned[col] = np.where(
                        (df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound),
                        df_cleaned[col].median(),
                        df_cleaned[col]
                    )
                    print(f"Handled {outliers} outliers in column: {col}")
        
        df_cleaned.to_csv(output_path, index=False)
        
        print(f"Cleaned data saved to: {output_path}")
        print(f"Final shape: {df_cleaned.shape}")
        print(f"Missing values after cleaning:")
        print(df_cleaned.isnull().sum())
        
        return df_cleaned
        
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df):
    """
    Validate the cleaned dataframe for common data quality issues.
    """
    if df is None:
        return False
    
    validation_results = {
        'has_duplicates': df.duplicated().sum() == 0,
        'has_nulls': df.isnull().sum().sum() == 0,
        'has_infinite': np.any(np.isinf(df.select_dtypes(include=[np.number]))),
        'row_count': len(df),
        'column_count': len(df.columns)
    }
    
    print("Data Validation Results:")
    for key, value in validation_results.items():
        print(f"{key}: {value}")
    
    return all([validation_results['has_duplicates'], 
                validation_results['has_nulls'] == 0,
                not validation_results['has_infinite']])

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = clean_csv_data(input_file, output_file)
    
    if cleaned_df is not None:
        is_valid = validate_dataframe(cleaned_df)
        if is_valid:
            print("Data cleaning completed successfully. Data is valid.")
        else:
            print("Data cleaning completed with warnings. Check validation results.")
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, threshold=1.5):
    """
    Remove outliers using IQR method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        threshold: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using min-max scaling.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return pd.Series([0.5] * len(data), index=data.index)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using z-score normalization.
    
    Args:
        data: pandas DataFrame
        column: column name to standardize
    
    Returns:
        Series with standardized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def detect_skewness(data, column):
    """
    Detect skewness in data.
    
    Args:
        data: pandas DataFrame
        column: column name to analyze
    
    Returns:
        Dictionary with skewness statistics
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    skew_val = stats.skew(data[column].dropna())
    
    result = {
        'skewness': skew_val,
        'skew_type': 'right_skewed' if skew_val > 0.5 else 
                    'left_skewed' if skew_val < -0.5 else 
                    'approximately_symmetric',
        'abs_skewness': abs(skew_val)
    }
    
    return result

def clean_dataset(data, numeric_columns=None, outlier_threshold=1.5):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric columns to clean (default: all numeric)
        outlier_threshold: IQR threshold for outlier removal
    
    Returns:
        Cleaned DataFrame
    """
    cleaned_data = data.copy()
    
    if numeric_columns is None:
        numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns.tolist()
    
    for column in numeric_columns:
        if column in cleaned_data.columns:
            # Remove outliers
            q1 = cleaned_data[column].quantile(0.25)
            q3 = cleaned_data[column].quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - outlier_threshold * iqr
            upper_bound = q3 + outlier_threshold * iqr
            
            mask = (cleaned_data[column] >= lower_bound) & (cleaned_data[column] <= upper_bound)
            cleaned_data = cleaned_data[mask]
    
    # Reset index after filtering
    cleaned_data = cleaned_data.reset_index(drop=True)
    
    return cleaned_data

def create_cleaning_report(data, cleaned_data, numeric_columns=None):
    """
    Generate a report comparing original and cleaned data.
    
    Args:
        data: original DataFrame
        cleaned_data: cleaned DataFrame
        numeric_columns: columns to include in report
    
    Returns:
        Dictionary with cleaning statistics
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    report = {
        'original_rows': len(data),
        'cleaned_rows': len(cleaned_data),
        'rows_removed': len(data) - len(cleaned_data),
        'removal_percentage': ((len(data) - len(cleaned_data)) / len(data)) * 100,
        'column_stats': {}
    }
    
    for column in numeric_columns:
        if column in data.columns:
            original_stats = {
                'mean': data[column].mean(),
                'std': data[column].std(),
                'min': data[column].min(),
                'max': data[column].max()
            }
            
            cleaned_stats = {
                'mean': cleaned_data[column].mean(),
                'std': cleaned_data[column].std(),
                'min': cleaned_data[column].min(),
                'max': cleaned_data[column].max()
            }
            
            report['column_stats'][column] = {
                'original': original_stats,
                'cleaned': cleaned_stats,
                'mean_change_percentage': ((cleaned_stats['mean'] - original_stats['mean']) / original_stats['mean']) * 100 if original_stats['mean'] != 0 else 0
            }
    
    return reportimport pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to remove duplicate rows. Default is True.
    fill_missing (str): Method to fill missing values. Options are 'mean', 'median', 'mode', or 'drop'. Default is 'mean'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median', 'mode']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        if fill_missing == 'mean':
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].mean())
        elif fill_missing == 'median':
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].median())
        elif fill_missing == 'mode':
            for col in numeric_cols:
                mode_val = cleaned_df[col].mode()
                if not mode_val.empty:
                    cleaned_df[col] = cleaned_df[col].fillna(mode_val.iloc[0])
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame for required columns and data types.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names. Default is None.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 3, None, 5],
        'B': [10, 20, 20, None, 50, 60],
        'C': ['x', 'y', 'y', 'z', 'z', 'x']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid = validate_dataframe(cleaned, required_columns=['A', 'B'])
    print(f"\nDataFrame validation: {is_valid}")
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Args:
        data (pd.DataFrame): Input DataFrame
        column (str): Column name to process
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    return filtered_data

def calculate_summary_statistics(data, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Args:
        data (pd.DataFrame): Input DataFrame
        column (str): Column name to analyze
    
    Returns:
        dict: Dictionary containing summary statistics
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': data[column].mean(),
        'median': data[column].median(),
        'std': data[column].std(),
        'min': data[column].min(),
        'max': data[column].max(),
        'count': data[column].count()
    }
    
    return stats

def validate_data_types(data, expected_types):
    """
    Validate that DataFrame columns match expected data types.
    
    Args:
        data (pd.DataFrame): Input DataFrame
        expected_types (dict): Dictionary mapping column names to expected dtypes
    
    Returns:
        bool: True if all columns match expected types, False otherwise
    """
    for column, expected_type in expected_types.items():
        if column in data.columns:
            if not np.issubdtype(data[column].dtype, expected_type):
                return False
    return True