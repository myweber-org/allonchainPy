
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

def calculate_basic_stats(df, column):
    """
    Calculate basic statistics for a DataFrame column.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to analyze
    
    Returns:
        dict: Dictionary containing statistics
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

def normalize_column(df, column):
    """
    Normalize a column using min-max scaling.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to normalize
    
    Returns:
        pd.DataFrame: DataFrame with normalized column
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_copy = df.copy()
    min_val = df_copy[column].min()
    max_val = df_copy[column].max()
    
    if max_val == min_val:
        df_copy[f'{column}_normalized'] = 0.5
    else:
        df_copy[f'{column}_normalized'] = (df_copy[column] - min_val) / (max_val - min_val)
    
    return df_copy

if __name__ == "__main__":
    sample_data = {
        'values': [10, 12, 12, 13, 12, 11, 14, 13, 15, 100, 12, 13, 12, 11, 10, 14, 13, 12, 11, 10]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    stats = calculate_basic_stats(df, 'values')
    print("Original Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")
    print()
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    print("Cleaned DataFrame (outliers removed):")
    print(cleaned_df)
    print()
    
    normalized_df = normalize_column(cleaned_df, 'values')
    print("DataFrame with normalized column:")
    print(normalized_df)
import pandas as pd
import numpy as np

def clean_dataframe(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop').
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            elif fill_missing == 'median':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            mode_val = cleaned_df[col].mode()
            if not mode_val.empty:
                cleaned_df[col] = cleaned_df[col].fillna(mode_val.iloc[0])
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

def remove_outliers(df, column, method='iqr', threshold=1.5):
    """
    Remove outliers from a specific column in DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column (str): Column name to process.
    method (str): Outlier detection method ('iqr' or 'zscore').
    threshold (float): Threshold for outlier detection.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' must be numeric")
    
    data = df[column].copy()
    
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        mask = (data >= lower_bound) & (data <= upper_bound)
    elif method == 'zscore':
        z_scores = np.abs((data - data.mean()) / data.std())
        mask = z_scores <= threshold
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")
    
    return df[mask]import pandas as pd
import numpy as np

def clean_csv_data(filepath, fill_strategy='mean', drop_threshold=0.5):
    """
    Load and clean CSV data by handling missing values and removing columns.
    
    Args:
        filepath: Path to the CSV file
        fill_strategy: Strategy for filling missing values ('mean', 'median', 'mode', 'zero')
        drop_threshold: Threshold for dropping columns with too many missing values (0.0-1.0)
    
    Returns:
        Cleaned DataFrame and cleaning report
    """
    try:
        df = pd.read_csv(filepath)
        original_shape = df.shape
        
        # Create cleaning report
        report = {
            'original_rows': original_shape[0],
            'original_columns': original_shape[1],
            'missing_values': df.isnull().sum().sum(),
            'cleaning_strategy': fill_strategy
        }
        
        # Drop columns with too many missing values
        missing_ratio = df.isnull().sum() / len(df)
        columns_to_drop = missing_ratio[missing_ratio > drop_threshold].index
        df = df.drop(columns=columns_to_drop)
        report['dropped_columns'] = len(columns_to_drop)
        
        # Fill missing values based on strategy
        if fill_strategy == 'mean':
            df = df.fillna(df.mean(numeric_only=True))
        elif fill_strategy == 'median':
            df = df.fillna(df.median(numeric_only=True))
        elif fill_strategy == 'mode':
            df = df.fillna(df.mode().iloc[0])
        elif fill_strategy == 'zero':
            df = df.fillna(0)
        else:
            raise ValueError(f"Unknown fill strategy: {fill_strategy}")
        
        # Remove duplicate rows
        duplicates = df.duplicated().sum()
        df = df.drop_duplicates()
        report['removed_duplicates'] = duplicates
        report['final_rows'] = df.shape[0]
        report['final_columns'] = df.shape[1]
        
        return df, report
        
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None, None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None, None

def validate_dataframe(df, required_columns=None, numeric_columns=None):
    """
    Validate DataFrame structure and data types.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        numeric_columns: List of columns that should be numeric
    
    Returns:
        Boolean indicating if validation passed and list of issues
    """
    issues = []
    
    if df is None or df.empty:
        issues.append("DataFrame is empty or None")
        return False, issues
    
    # Check required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")
    
    # Check numeric columns
    if numeric_columns:
        for col in numeric_columns:
            if col in df.columns:
                if not np.issubdtype(df[col].dtype, np.number):
                    issues.append(f"Column '{col}' is not numeric")
    
    # Check for infinite values
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        infinite_count = np.isinf(numeric_df.values).sum()
        if infinite_count > 0:
            issues.append(f"Found {infinite_count} infinite values in numeric columns")
    
    return len(issues) == 0, issues

def save_cleaned_data(df, output_path, format='csv'):
    """
    Save cleaned DataFrame to file.
    
    Args:
        df: Cleaned DataFrame
        output_path: Path to save the cleaned data
        format: Output format ('csv', 'parquet', 'json')
    
    Returns:
        Boolean indicating success
    """
    try:
        if format == 'csv':
            df.to_csv(output_path, index=False)
        elif format == 'parquet':
            df.to_parquet(output_path, index=False)
        elif format == 'json':
            df.to_json(output_path, orient='records')
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Cleaned data saved to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error saving data: {str(e)}")
        return False

# Example usage
if __name__ == "__main__":
    # Clean the data
    cleaned_df, report = clean_csv_data('raw_data.csv', fill_strategy='median')
    
    if cleaned_df is not None:
        print(f"Cleaning report: {report}")
        
        # Validate the cleaned data
        is_valid, issues = validate_dataframe(cleaned_df)
        
        if is_valid:
            print("Data validation passed")
            # Save the cleaned data
            save_cleaned_data(cleaned_df, 'cleaned_data.csv')
        else:
            print(f"Data validation failed: {issues}")