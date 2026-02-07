
import pandas as pd
import numpy as np

def clean_dataset(df, missing_strategy='mean', outlier_method='iqr'):
    """
    Clean a pandas DataFrame by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    missing_strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
    outlier_method (str): Method for handling outliers ('iqr', 'zscore')
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    
    cleaned_df = df.copy()
    
    # Handle missing values
    if missing_strategy == 'mean':
        cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
    elif missing_strategy == 'median':
        cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
    elif missing_strategy == 'mode':
        cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
    elif missing_strategy == 'drop':
        cleaned_df = cleaned_df.dropna()
    
    # Handle outliers for numeric columns only
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    
    if outlier_method == 'iqr':
        for col in numeric_cols:
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers
            cleaned_df[col] = cleaned_df[col].clip(lower=lower_bound, upper=upper_bound)
    
    elif outlier_method == 'zscore':
        for col in numeric_cols:
            z_scores = np.abs((cleaned_df[col] - cleaned_df[col].mean()) / cleaned_df[col].std())
            mask = z_scores > 3
            cleaned_df.loc[mask, col] = cleaned_df[col].mean()
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the structure and content of a DataFrame.
    
    Parameters:
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
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Data validation passed"

# Example usage
if __name__ == "__main__":
    # Create sample data with missing values and outliers
    sample_data = {
        'A': [1, 2, np.nan, 4, 100],
        'B': [5, 6, 7, np.nan, 8],
        'C': [10, 20, 30, 40, 50]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    # Clean the data
    cleaned_df = clean_dataset(df, missing_strategy='mean', outlier_method='iqr')
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\n")
    
    # Validate the cleaned data
    is_valid, message = validate_data(cleaned_df, required_columns=['A', 'B', 'C'], min_rows=3)
    print(f"Validation result: {is_valid}")
    print(f"Validation message: {message}")import pandas as pd
import numpy as np

def clean_dataset(df, missing_strategy='mean', outlier_method='iqr', columns=None):
    """
    Clean a pandas DataFrame by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    missing_strategy (str): Strategy for missing values - 'mean', 'median', 'mode', or 'drop'
    outlier_method (str): Method for outlier detection - 'iqr' or 'zscore'
    columns (list): Specific columns to clean, if None clean all numeric columns
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if columns is None:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        if df_clean[col].dtype in [np.float64, np.int64]:
            handle_missing_values(df_clean, col, missing_strategy)
            handle_outliers(df_clean, col, outlier_method)
    
    return df_clean

def handle_missing_values(df, column, strategy='mean'):
    """Handle missing values in a specific column."""
    if strategy == 'mean':
        fill_value = df[column].mean()
    elif strategy == 'median':
        fill_value = df[column].median()
    elif strategy == 'mode':
        fill_value = df[column].mode()[0] if not df[column].mode().empty else 0
    elif strategy == 'drop':
        df.dropna(subset=[column], inplace=True)
        return
    else:
        raise ValueError(f"Unknown missing value strategy: {strategy}")
    
    df[column].fillna(fill_value, inplace=True)

def handle_outliers(df, column, method='iqr'):
    """Handle outliers in a specific column using IQR or Z-score method."""
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        median_val = df[column].median()
        df[column] = np.where((df[column] < lower_bound) | (df[column] > upper_bound), 
                              median_val, df[column])
    
    elif method == 'zscore':
        mean_val = df[column].mean()
        std_val = df[column].std()
        z_scores = np.abs((df[column] - mean_val) / std_val)
        
        threshold = 3
        median_val = df[column].median()
        df[column] = np.where(z_scores > threshold, median_val, df[column])
    
    else:
        raise ValueError(f"Unknown outlier method: {method}")

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and data quality.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of columns that must be present
    
    Returns:
    dict: Dictionary with validation results
    """
    validation_results = {
        'is_valid': True,
        'missing_columns': [],
        'empty_dataframe': False,
        'null_percentage': {}
    }
    
    if df.empty:
        validation_results['is_valid'] = False
        validation_results['empty_dataframe'] = True
        return validation_results
    
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            validation_results['is_valid'] = False
            validation_results['missing_columns'] = missing
    
    for col in df.columns:
        null_count = df[col].isnull().sum()
        total_count = len(df[col])
        null_pct = (null_count / total_count * 100) if total_count > 0 else 0
        validation_results['null_percentage'][col] = round(null_pct, 2)
    
    return validation_results

def create_sample_data():
    """Create sample data for testing the cleaning functions."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'age': np.random.normal(35, 10, n_samples),
        'salary': np.random.normal(50000, 15000, n_samples),
        'score': np.random.normal(75, 15, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    df.loc[np.random.choice(n_samples, 10, replace=False), 'age'] = np.nan
    df.loc[np.random.choice(n_samples, 5, replace=False), 'salary'] = np.nan
    
    df.loc[0, 'salary'] = 1000000
    df.loc[1, 'score'] = -50
    df.loc[2, 'age'] = 150
    
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    print("Original Data Sample:")
    print(sample_df.head())
    print("\nValidation Results:")
    print(validate_dataframe(sample_df))
    
    cleaned_df = clean_dataset(sample_df, missing_strategy='median', outlier_method='iqr')
    print("\nCleaned Data Sample:")
    print(cleaned_df.head())
    print("\nCleaned Validation Results:")
    print(validate_dataframe(cleaned_df))