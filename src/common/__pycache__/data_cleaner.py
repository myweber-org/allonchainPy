
import pandas as pd
import numpy as np

def clean_dataframe(df, drop_duplicates=True, fill_missing=True):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to remove duplicate rows
        fill_missing (bool): Whether to fill missing values with column means
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if cleaned_df[col].isnull().any():
                mean_val = cleaned_df[col].mean()
                cleaned_df[col] = cleaned_df[col].fillna(mean_val)
                print(f"Filled missing values in column '{col}' with mean: {mean_val:.2f}")
    
    return cleaned_df

def validate_dataframe(df):
    """
    Validate DataFrame for common data quality issues.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
    
    Returns:
        dict: Dictionary containing validation results
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(df.select_dtypes(include=['object']).columns)
    }
    
    return validation_results

def save_cleaned_data(df, output_path):
    """
    Save cleaned DataFrame to CSV file.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        output_path (str): Path for output CSV file
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'value': [10.5, 20.3, 20.3, None, 40.1, 50.0],
        'category': ['A', 'B', 'B', 'C', 'A', 'B']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nValidation Results:")
    print(validate_dataframe(df))
    
    cleaned = clean_dataframe(df)
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    save_cleaned_data(cleaned, 'cleaned_data.csv')
import pandas as pd
import numpy as np

def clean_dataset(df, missing_strategy='mean', outlier_method='iqr', columns=None):
    """
    Clean a pandas DataFrame by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    missing_strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
    outlier_method (str): Method for detecting outliers ('iqr', 'zscore')
    columns (list): Specific columns to clean, if None clean all numeric columns
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    # Determine columns to process
    if columns is None:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    # Handle missing values
    for col in columns:
        if col in df_clean.columns:
            if missing_strategy == 'mean':
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
            elif missing_strategy == 'median':
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            elif missing_strategy == 'mode':
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
            elif missing_strategy == 'drop':
                df_clean.dropna(subset=[col], inplace=True)
    
    # Handle outliers
    for col in columns:
        if col in df_clean.columns:
            if outlier_method == 'iqr':
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers
                df_clean[col] = np.where(df_clean[col] < lower_bound, lower_bound, df_clean[col])
                df_clean[col] = np.where(df_clean[col] > upper_bound, upper_bound, df_clean[col])
            
            elif outlier_method == 'zscore':
                mean_val = df_clean[col].mean()
                std_val = df_clean[col].std()
                z_scores = np.abs((df_clean[col] - mean_val) / std_val)
                
                # Remove rows with z-score > 3
                df_clean = df_clean[z_scores <= 3]
    
    return df_clean

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): Columns that must be present
    min_rows (int): Minimum number of rows required
    
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
    
    return True, "Data validation passed"

def normalize_data(df, columns=None, method='minmax'):
    """
    Normalize numeric columns in DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): Columns to normalize, if None normalize all numeric columns
    method (str): Normalization method ('minmax' or 'standard')
    
    Returns:
    pd.DataFrame: Normalized DataFrame
    """
    df_norm = df.copy()
    
    if columns is None:
        numeric_cols = df_norm.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    for col in columns:
        if col in df_norm.columns:
            if method == 'minmax':
                min_val = df_norm[col].min()
                max_val = df_norm[col].max()
                if max_val > min_val:  # Avoid division by zero
                    df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
            
            elif method == 'standard':
                mean_val = df_norm[col].mean()
                std_val = df_norm[col].std()
                if std_val > 0:  # Avoid division by zero
                    df_norm[col] = (df_norm[col] - mean_val) / std_val
    
    return df_norm

# Example usage
if __name__ == "__main__":
    # Create sample data
    data = {
        'A': [1, 2, np.nan, 4, 100],
        'B': [5, 6, 7, 8, 9],
        'C': [10, 11, 12, 13, 14]
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    
    # Clean the data
    df_clean = clean_dataset(df, missing_strategy='mean', outlier_method='iqr')
    print("\nCleaned DataFrame:")
    print(df_clean)
    
    # Validate data
    is_valid, message = validate_data(df_clean, required_columns=['A', 'B', 'C'])
    print(f"\nValidation: {is_valid} - {message}")
    
    # Normalize data
    df_norm = normalize_data(df_clean, method='minmax')
    print("\nNormalized DataFrame:")
    print(df_norm)import pandas as pd
import numpy as np

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): Strategy to handle missing values ('mean', 'median', 'mode', 'drop')
    columns (list): List of columns to apply the strategy to, if None applies to all numeric columns
    
    Returns:
    pd.DataFrame: DataFrame with handled missing values
    """
    df_copy = df.copy()
    
    if columns is None:
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    if strategy == 'drop':
        return df_copy.dropna(subset=columns)
    
    for col in columns:
        if col in df_copy.columns:
            if strategy == 'mean':
                fill_value = df_copy[col].mean()
            elif strategy == 'median':
                fill_value = df_copy[col].median()
            elif strategy == 'mode':
                fill_value = df_copy[col].mode()[0] if not df_copy[col].mode().empty else np.nan
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            df_copy[col] = df_copy[col].fillna(fill_value)
    
    return df_copy

def remove_outliers_iqr(df, columns=None, threshold=1.5):
    """
    Remove outliers using the Interquartile Range (IQR) method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of columns to check for outliers, if None checks all numeric columns
    threshold (float): IQR multiplier for outlier detection
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    df_copy = df.copy()
    
    if columns is None:
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    outlier_mask = pd.Series([False] * len(df_copy))
    
    for col in columns:
        if col in df_copy.columns:
            Q1 = df_copy[col].quantile(0.25)
            Q3 = df_copy[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            col_outliers = (df_copy[col] < lower_bound) | (df_copy[col] > upper_bound)
            outlier_mask = outlier_mask | col_outliers
    
    return df_copy[~outlier_mask].reset_index(drop=True)

def standardize_columns(df, columns=None):
    """
    Standardize numeric columns to have zero mean and unit variance.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of columns to standardize, if None standardizes all numeric columns
    
    Returns:
    pd.DataFrame: DataFrame with standardized columns
    """
    df_copy = df.copy()
    
    if columns is None:
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    for col in columns:
        if col in df_copy.columns:
            mean_val = df_copy[col].mean()
            std_val = df_copy[col].std()
            
            if std_val > 0:
                df_copy[col] = (df_copy[col] - mean_val) / std_val
    
    return df_copy

def clean_dataset(df, missing_strategy='mean', outlier_threshold=1.5, standardize=True):
    """
    Comprehensive data cleaning pipeline.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    missing_strategy (str): Strategy for handling missing values
    outlier_threshold (float): IQR threshold for outlier removal
    standardize (bool): Whether to standardize numeric columns
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = handle_missing_values(df, strategy=missing_strategy)
    cleaned_df = remove_outliers_iqr(cleaned_df, threshold=outlier_threshold)
    
    if standardize:
        cleaned_df = standardize_columns(cleaned_df)
    
    return cleaned_df