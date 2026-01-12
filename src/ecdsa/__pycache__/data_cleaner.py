import pandas as pd
import numpy as np

def clean_dataset(df, missing_strategy='mean', outlier_method='iqr', columns=None):
    """
    Clean dataset by handling missing values and outliers.
    
    Args:
        df (pd.DataFrame): Input dataframe
        missing_strategy (str): Strategy for missing values ('mean', 'median', 'mode', 'drop')
        outlier_method (str): Method for outlier detection ('iqr', 'zscore')
        columns (list): Specific columns to clean, defaults to all numeric columns
    
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        if missing_strategy != 'drop':
            if missing_strategy == 'mean':
                fill_value = df_clean[col].mean()
            elif missing_strategy == 'median':
                fill_value = df_clean[col].median()
            elif missing_strategy == 'mode':
                fill_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 0
            else:
                fill_value = 0
                
            df_clean[col].fillna(fill_value, inplace=True)
        else:
            df_clean = df_clean.dropna(subset=[col])
        
        if outlier_method == 'iqr':
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            if missing_strategy == 'drop':
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
            else:
                df_clean.loc[df_clean[col] < lower_bound, col] = lower_bound
                df_clean.loc[df_clean[col] > upper_bound, col] = upper_bound
                
        elif outlier_method == 'zscore':
            mean_val = df_clean[col].mean()
            std_val = df_clean[col].std()
            
            if std_val > 0:
                z_scores = np.abs((df_clean[col] - mean_val) / std_val)
                
                if missing_strategy == 'drop':
                    df_clean = df_clean[z_scores <= 3]
                else:
                    df_clean.loc[z_scores > 3, col] = mean_val
    
    return df_clean

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate dataframe structure and content.
    
    Args:
        df (pd.DataFrame): Dataframe to validate
        required_columns (list): Columns that must be present
        min_rows (int): Minimum number of rows required
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(df) < min_rows:
        return False, f"Dataframe has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Dataframe is valid"

def get_data_summary(df):
    """
    Generate summary statistics for dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        dict: Summary statistics
    """
    summary = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_summary': {},
        'categorical_summary': {}
    }
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        summary['numeric_summary'][col] = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'median': df[col].median()
        }
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        summary['categorical_summary'][col] = {
            'unique_count': df[col].nunique(),
            'top_value': df[col].mode()[0] if not df[col].mode().empty else None,
            'top_count': df[col].value_counts().iloc[0] if not df[col].value_counts().empty else 0
        }
    
    return summary

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 100],
        'B': [5, 6, 7, 8, 9],
        'C': ['x', 'y', 'z', 'x', 'y']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nSummary:")
    print(get_data_summary(df))
    
    cleaned_df = clean_dataset(df, missing_strategy='mean', outlier_method='iqr')
    print("\nCleaned DataFrame:")
    print(cleaned_df)import pandas as pd

def clean_dataset(df, remove_duplicates=True, fill_method='drop'):
    """
    Clean a pandas DataFrame by handling missing values and optionally removing duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    remove_duplicates (bool): If True, remove duplicate rows.
    fill_method (str): Method to handle missing values: 'drop' to remove rows, 
                       'mean' to fill with column mean (numeric only), 
                       'median' to fill with column median (numeric only),
                       'mode' to fill with most frequent value.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if fill_method == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_method in ['mean', 'median', 'mode']:
        for column in cleaned_df.columns:
            if cleaned_df[column].dtype in ['int64', 'float64']:
                if fill_method == 'mean':
                    cleaned_df[column].fillna(cleaned_df[column].mean(), inplace=True)
                elif fill_method == 'median':
                    cleaned_df[column].fillna(cleaned_df[column].median(), inplace=True)
                elif fill_method == 'mode':
                    cleaned_df[column].fillna(cleaned_df[column].mode()[0], inplace=True)
            else:
                # For non-numeric columns, use mode or drop
                if fill_method == 'mode':
                    cleaned_df[column].fillna(cleaned_df[column].mode()[0], inplace=True)
                else:
                    cleaned_df = cleaned_df.dropna(subset=[column])
    
    # Remove duplicates if requested
    if remove_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Reset index after cleaning
    cleaned_df.reset_index(drop=True, inplace=True)
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate a DataFrame for basic integrity checks.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    dict: Dictionary with validation results.
    """
    validation_results = {
        'is_valid': True,
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'column_info': {}
    }
    
    # Check required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['missing_required_columns'] = missing_columns
    
    # Collect column information
    for column in df.columns:
        validation_results['column_info'][column] = {
            'dtype': str(df[column].dtype),
            'unique_values': df[column].nunique(),
            'missing_values': df[column].isnull().sum()
        }
    
    return validation_results

# Example usage (commented out for production)
# if __name__ == "__main__":
#     # Create sample data
#     sample_data = {
#         'A': [1, 2, None, 4, 5, 5],
#         'B': [10, 20, 30, None, 50, 50],
#         'C': ['x', 'y', 'z', 'x', None, 'x']
#     }
#     
#     df = pd.DataFrame(sample_data)
#     print("Original DataFrame:")
#     print(df)
#     
#     # Clean the data
#     cleaned = clean_dataset(df, remove_duplicates=True, fill_method='mean')
#     print("\nCleaned DataFrame:")
#     print(cleaned)
#     
#     # Validate the cleaned data
#     validation = validate_dataset(cleaned)
#     print("\nValidation Results:")
#     print(validation)
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, convert_types=True):
    """
    Clean a pandas DataFrame by removing duplicates and converting data types.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    drop_duplicates (bool): Whether to remove duplicate rows
    convert_types (bool): Whether to convert columns to optimal data types
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if convert_types:
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object':
                try:
                    cleaned_df[col] = pd.to_datetime(cleaned_df[col])
                    print(f"Converted column '{col}' to datetime")
                except (ValueError, TypeError):
                    try:
                        cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='ignore')
                        if cleaned_df[col].dtype != 'object':
                            print(f"Converted column '{col}' to numeric")
                    except (ValueError, TypeError):
                        pass
    
    cleaned_df = cleaned_df.reset_index(drop=True)
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
    
    if df.empty:
        validation_results['is_valid'] = False
        validation_results['errors'].append('DataFrame is empty')
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f'Missing required columns: {missing_columns}')
    
    for col in df.columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            validation_results['warnings'].append(f"Column '{col}' has {null_count} null values")
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'date': ['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02'],
        'value': ['100', '100', '200', '300'],
        'category': ['A', 'A', 'B', 'C']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nDataFrame info:")
    print(df.info())
    
    cleaned = clean_dataset(df)
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    validation = validate_dataframe(cleaned, required_columns=['date', 'value'])
    print("\nValidation results:")
    print(validation)import numpy as np
import pandas as pd

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
    if max_val == min_val:
        return df[column].apply(lambda x: 0.5)
    return df[column].apply(lambda x: (x - min_val) / (max_val - min_val))

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
    return cleaned_df.reset_index(drop=True)

def validate_data(df, required_columns):
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    return True