
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to process
    factor (float): Multiplier for IQR (default 1.5)
    
    Returns:
    pd.DataFrame: Dataframe with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data.copy()

def normalize_minmax(data, column):
    """
    Normalize data to [0, 1] range using min-max scaling.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to normalize
    
    Returns:
    pd.DataFrame: Dataframe with normalized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    result = data.copy()
    min_val = result[column].min()
    max_val = result[column].max()
    
    if max_val == min_val:
        result[column] = 0.5
    else:
        result[column] = (result[column] - min_val) / (max_val - min_val)
    
    return result

def standardize_zscore(data, column):
    """
    Standardize data using z-score normalization.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to standardize
    
    Returns:
    pd.DataFrame: Dataframe with standardized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    result = data.copy()
    mean_val = result[column].mean()
    std_val = result[column].std()
    
    if std_val == 0:
        result[column] = 0
    else:
        result[column] = (result[column] - mean_val) / std_val
    
    return result

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in specified columns.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    strategy (str): Imputation strategy ('mean', 'median', 'mode', 'drop')
    columns (list): List of columns to process, None for all numeric columns
    
    Returns:
    pd.DataFrame: Dataframe with handled missing values
    """
    result = data.copy()
    
    if columns is None:
        columns = result.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if col not in result.columns:
            continue
            
        if strategy == 'drop':
            result = result.dropna(subset=[col])
        elif strategy == 'mean':
            result[col] = result[col].fillna(result[col].mean())
        elif strategy == 'median':
            result[col] = result[col].fillna(result[col].median())
        elif strategy == 'mode':
            result[col] = result[col].fillna(result[col].mode()[0])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    return result

def create_data_summary(data):
    """
    Create a summary statistics dataframe.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    
    Returns:
    pd.DataFrame: Summary statistics
    """
    summary = pd.DataFrame({
        'column': data.columns,
        'dtype': data.dtypes.values,
        'non_null_count': data.count().values,
        'null_count': data.isnull().sum().values,
        'null_percentage': (data.isnull().sum() / len(data) * 100).values,
        'unique_count': data.nunique().values
    })
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        numeric_stats = data[numeric_cols].describe().T
        summary = summary.merge(
            numeric_stats[['mean', 'std', 'min', '25%', '50%', '75%', 'max']],
            left_on='column',
            right_index=True,
            how='left'
        )
    
    return summary.reset_index(drop=True)
import pandas as pd

def clean_dataframe(df, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to remove duplicate rows
        fill_missing (str or dict): Method to fill missing values:
            - 'mean': Fill with column mean
            - 'median': Fill with column median
            - 'mode': Fill with column mode
            - dict: Column-specific fill values
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing is not None:
        if fill_missing == 'mean':
            cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
        elif fill_missing == 'median':
            cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
        elif fill_missing == 'mode':
            cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
        elif isinstance(fill_missing, dict):
            cleaned_df = cleaned_df.fillna(fill_missing)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
        min_rows (int): Minimum number of rows required
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"

def remove_outliers_iqr(df, columns, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of column names to check for outliers
        multiplier (float): IQR multiplier for outlier detection
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    df_clean = df.copy()
    
    for column in columns:
        if column in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[column]):
            Q1 = df_clean[column].quantile(0.25)
            Q3 = df_clean[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            df_clean = df_clean[(df_clean[column] >= lower_bound) & 
                               (df_clean[column] <= upper_bound)]
    
    return df_clean
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_strategy='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to remove duplicate rows
        fill_missing (bool): Whether to fill missing values
        fill_strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', 'zero')
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    # Remove duplicate rows
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    # Handle missing values
    if fill_missing:
        for column in cleaned_df.columns:
            if cleaned_df[column].dtype in ['int64', 'float64']:
                if fill_strategy == 'mean':
                    fill_value = cleaned_df[column].mean()
                elif fill_strategy == 'median':
                    fill_value = cleaned_df[column].median()
                elif fill_strategy == 'mode':
                    fill_value = cleaned_df[column].mode()[0] if not cleaned_df[column].mode().empty else 0
                elif fill_strategy == 'zero':
                    fill_value = 0
                else:
                    fill_value = cleaned_df[column].mean()
                
                missing_count = cleaned_df[column].isna().sum()
                if missing_count > 0:
                    cleaned_df[column] = cleaned_df[column].fillna(fill_value)
                    print(f"Filled {missing_count} missing values in column '{column}' with {fill_strategy}: {fill_value}")
    
    return cleaned_df

def validate_dataset(df, check_duplicates=True, check_missing=True):
    """
    Validate a DataFrame for data quality issues.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        check_duplicates (bool): Check for duplicate rows
        check_missing (bool): Check for missing values
    
    Returns:
        dict: Dictionary containing validation results
    """
    validation_results = {}
    
    if check_duplicates:
        duplicate_count = df.duplicated().sum()
        validation_results['duplicate_rows'] = duplicate_count
    
    if check_missing:
        missing_values = df.isna().sum().to_dict()
        validation_results['missing_values'] = missing_values
        total_missing = sum(missing_values.values())
        validation_results['total_missing'] = total_missing
    
    return validation_results

# Example usage
if __name__ == "__main__":
    # Create sample data with duplicates and missing values
    sample_data = {
        'id': [1, 2, 3, 1, 5, 6, 7, 8],
        'value': [10.5, np.nan, 15.2, 10.5, 20.1, np.nan, 25.3, 30.0],
        'category': ['A', 'B', 'A', 'A', 'C', 'B', 'A', 'C']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Validate before cleaning
    print("Validation before cleaning:")
    validation = validate_dataset(df)
    for key, value in validation.items():
        print(f"{key}: {value}")
    
    print("\n" + "="*50 + "\n")
    
    # Clean the dataset
    cleaned_df = clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_strategy='mean')
    
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    print("\n" + "="*50 + "\n")
    
    # Validate after cleaning
    print("Validation after cleaning:")
    validation_after = validate_dataset(cleaned_df)
    for key, value in validation_after.items():
        print(f"{key}: {value}")import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range (IQR) method.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The column name to clean.
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed from the specified column.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df.reset_index(drop=True)

def clean_dataset(df, numeric_columns=None):
    """
    Clean a dataset by removing outliers from all numeric columns.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        numeric_columns (list, optional): List of numeric column names. 
                                         If None, uses all numeric columns.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame with outliers removed.
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in df.columns:
            original_len = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            removed_count = original_len - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{col}'")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'id': range(1, 101),
        'value': np.concatenate([
            np.random.normal(100, 10, 90),
            np.array([500, -300, 1000, 800, -500])
        ]),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(sample_data)
    print(f"Original dataset shape: {df.shape}")
    
    cleaned_df = clean_dataset(df, ['value'])
    print(f"Cleaned dataset shape: {cleaned_df.shape}")
    print(f"Outliers removed: {len(df) - len(cleaned_df)}")
import pandas as pd
import numpy as np

def clean_dataframe(df, drop_na=True, rename_columns=True):
    """
    Clean a pandas DataFrame by handling missing values and standardizing column names.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    drop_na (bool): Whether to drop rows with any null values
    rename_columns (bool): Whether to standardize column names
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if drop_na:
        df_clean = df_clean.dropna()
    
    if rename_columns:
        df_clean.columns = (
            df_clean.columns
            .str.strip()
            .str.lower()
            .str.replace(' ', '_')
            .str.replace(r'[^\w_]', '', regex=True)
        )
    
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
    
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    dict: Validation results with status and messages
    """
    validation_result = {
        'is_valid': True,
        'messages': [],
        'row_count': len(df),
        'column_count': len(df.columns)
    }
    
    if df.empty:
        validation_result['is_valid'] = False
        validation_result['messages'].append('DataFrame is empty')
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_result['is_valid'] = False
            validation_result['messages'].append(f'Missing required columns: {missing_columns}')
    
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        validation_result['messages'].append(f'Found {null_counts.sum()} null values across columns')
    
    return validation_result

def sample_data():
    """
    Create sample DataFrame for testing.
    
    Returns:
    pd.DataFrame: Sample DataFrame with mixed data
    """
    data = {
        'User ID': [1, 2, 3, 4, 5],
        'User Name': ['Alice', 'Bob', None, 'David', 'Eve'],
        'Age': [25, 30, 35, None, 28],
        'Score': [85.5, 92.0, 78.5, 88.0, 95.5],
        'Join Date': ['2023-01-15', '2023-02-20', '2023-03-10', None, '2023-05-05']
    }
    return pd.DataFrame(data)

if __name__ == '__main__':
    df = sample_data()
    print("Original DataFrame:")
    print(df)
    print("\nDataFrame Info:")
    print(df.info())
    
    cleaned_df = clean_dataframe(df, drop_na=True, rename_columns=True)
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    validation = validate_dataframe(cleaned_df, required_columns=['user_id', 'user_name', 'age'])
    print("\nValidation Results:")
    for key, value in validation.items():
        print(f"{key}: {value}")