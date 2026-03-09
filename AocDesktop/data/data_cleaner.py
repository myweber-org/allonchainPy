
import pandas as pd
import numpy as np

def remove_outliers(df, column, threshold=3):
    mean = df[column].mean()
    std = df[column].std()
    z_scores = np.abs((df[column] - mean) / std)
    return df[z_scores < threshold]

def normalize_column(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(file_path):
    df = pd.read_csv(file_path)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        df = remove_outliers(df, col)
        df = normalize_column(df, col)
    
    df = df.dropna()
    
    output_path = file_path.replace('.csv', '_cleaned.csv')
    df.to_csv(output_path, index=False)
    return output_path

if __name__ == "__main__":
    cleaned_file = clean_dataset('raw_data.csv')
    print(f"Cleaned data saved to: {cleaned_file}")
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    original_shape = df.shape
    
    if drop_duplicates:
        df = df.drop_duplicates()
        print(f"Removed {original_shape[0] - df.shape[0]} duplicate rows")
    
    if fill_missing:
        if fill_missing == 'mean':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif fill_missing == 'median':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif fill_missing == 'mode':
            for col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else None)
        elif fill_missing == 'drop':
            df = df.dropna()
    
    print(f"Original shape: {original_shape}")
    print(f"Cleaned shape: {df.shape}")
    
    return df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the dataset meets basic requirements.
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    if len(df) < min_rows:
        raise ValueError(f"Dataset must have at least {min_rows} rows, but has {len(df)}")
    
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'value': [10, 20, 20, np.nan, 40, 50],
        'category': ['A', 'B', 'B', 'C', None, 'A']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_dataset(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    try:
        validate_data(cleaned_df, required_columns=['id', 'value'], min_rows=3)
        print("Data validation passed")
    except ValueError as e:
        print(f"Data validation failed: {e}")
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, handle_nulls='drop', fill_value=None):
    """
    Clean a pandas DataFrame by handling duplicates and null values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    drop_duplicates (bool): Whether to drop duplicate rows
    handle_nulls (str): How to handle nulls - 'drop', 'fill', or 'ignore'
    fill_value: Value to fill nulls with if handle_nulls='fill'
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if handle_nulls == 'drop':
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.dropna()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} rows with null values")
    elif handle_nulls == 'fill' and fill_value is not None:
        cleaned_df = cleaned_df.fillna(fill_value)
        print(f"Filled null values with {fill_value}")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    if not isinstance(df, pd.DataFrame):
        validation_results['is_valid'] = False
        validation_results['errors'].append('Input is not a pandas DataFrame')
        return validation_results
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f'Missing required columns: {missing_columns}')
    
    if df.empty:
        validation_results['warnings'].append('DataFrame is empty')
    
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        validation_results['warnings'].append(f'Found {null_counts.sum()} null values in DataFrame')
    
    return validation_results

def sample_data_processing():
    """
    Example usage of the data cleaning functions.
    """
    data = {
        'id': [1, 2, 3, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'Charlie', None, 'Eve'],
        'age': [25, 30, None, 35, 28, 22],
        'score': [85.5, 92.0, 78.5, 78.5, 88.0, 95.5]
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    validation = validate_dataframe(df, required_columns=['id', 'name', 'age'])
    print("Validation Results:")
    print(validation)
    print("\n" + "="*50 + "\n")
    
    cleaned_df = clean_dataset(
        df, 
        drop_duplicates=True, 
        handle_nulls='fill', 
        fill_value={'name': 'Unknown', 'age': df['age'].mean()}
    )
    
    print("Cleaned DataFrame:")
    print(cleaned_df)
    
    return cleaned_df

if __name__ == "__main__":
    result_df = sample_data_processing()
    print(f"\nFinal DataFrame shape: {result_df.shape}")
import numpy as np

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_dataimport numpy as np
import pandas as pd
from scipy import stats

def detect_outliers_iqr(data, threshold=1.5):
    """
    Detect outliers using Interquartile Range method.
    Returns boolean mask where True indicates an outlier.
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    return (data < lower_bound) | (data > upper_bound)

def remove_outliers(df, columns, method='iqr', **kwargs):
    """
    Remove outliers from specified columns in DataFrame.
    Supports 'iqr' and 'zscore' methods.
    """
    df_clean = df.copy()
    
    for col in columns:
        if method == 'iqr':
            outlier_mask = detect_outliers_iqr(df[col], **kwargs)
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outlier_mask = z_scores > kwargs.get('threshold', 3)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        df_clean = df_clean[~outlier_mask]
    
    return df_clean

def normalize_data(df, columns, method='minmax'):
    """
    Normalize specified columns in DataFrame.
    Supports 'minmax' and 'standard' normalization.
    """
    df_normalized = df.copy()
    
    for col in columns:
        if method == 'minmax':
            min_val = df[col].min()
            max_val = df[col].max()
            df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
        elif method == 'standard':
            mean_val = df[col].mean()
            std_val = df[col].std()
            df_normalized[col] = (df[col] - mean_val) / std_val
        else:
            raise ValueError(f"Unsupported method: {method}")
    
    return df_normalized

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize_method='standard'):
    """
    Comprehensive data cleaning pipeline.
    Removes outliers and normalizes numeric columns.
    """
    # Remove outliers
    df_clean = remove_outliers(df, numeric_columns, method=outlier_method)
    
    # Normalize data
    df_normalized = normalize_data(df_clean, numeric_columns, method=normalize_method)
    
    # Reset index after cleaning
    df_normalized = df_normalized.reset_index(drop=True)
    
    return df_normalized

def validate_data(df, required_columns, min_rows=10):
    """
    Validate dataset meets minimum requirements.
    Returns tuple of (is_valid, error_message).
    """
    if len(df) < min_rows:
        return False, f"Dataset has fewer than {min_rows} rows"
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return False, f"Missing required columns: {missing_columns}"
    
    null_counts = df[required_columns].isnull().sum()
    if null_counts.any():
        return False, f"Columns with null values: {null_counts[null_counts > 0].to_dict()}"
    
    return True, "Dataset validation passed"