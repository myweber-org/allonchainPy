
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing=True, strategy='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to drop duplicate rows
        fill_missing (bool): Whether to fill missing values
        strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', 'zero')
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing and cleaned_df.isnull().any().any():
        for column in cleaned_df.columns:
            if cleaned_df[column].dtype in ['int64', 'float64']:
                if strategy == 'mean':
                    fill_value = cleaned_df[column].mean()
                elif strategy == 'median':
                    fill_value = cleaned_df[column].median()
                elif strategy == 'zero':
                    fill_value = 0
                else:
                    fill_value = cleaned_df[column].mode()[0] if not cleaned_df[column].mode().empty else 0
                
                missing_count = cleaned_df[column].isnull().sum()
                if missing_count > 0:
                    cleaned_df[column] = cleaned_df[column].fillna(fill_value)
                    print(f"Filled {missing_count} missing values in column '{column}' with {strategy}: {fill_value}")
            else:
                cleaned_df[column] = cleaned_df[column].fillna('Unknown')
    
    return cleaned_df

def validate_dataset(df, required_columns=None, min_rows=1):
    """
    Validate dataset structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
        min_rows (int): Minimum number of rows required
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if len(df) < min_rows:
        print(f"Dataset has only {len(df)} rows, minimum required is {min_rows}")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    return True

def remove_outliers(df, column, method='iqr', threshold=1.5):
    """
    Remove outliers from a specific column using specified method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to check for outliers
        method (str): Method for outlier detection ('iqr' or 'zscore')
        threshold (float): Threshold for outlier detection
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        print(f"Column '{column}' not found in DataFrame")
        return df
    
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        mask = (df[column] >= lower_bound) & (df[column] <= upper_bound)
    else:
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        mask = z_scores < threshold
    
    outliers_removed = len(df) - mask.sum()
    if outliers_removed > 0:
        print(f"Removed {outliers_removed} outliers from column '{column}'")
    
    return df[mask]

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 3, 4, 5, 6],
        'value': [10.5, 20.3, None, 15.7, 30.1, None, 1000.0],
        'category': ['A', 'B', 'C', 'C', 'A', None, 'D']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\nDataset info:")
    print(df.info())
    
    cleaned = clean_dataset(df, strategy='median')
    print("\nCleaned dataset:")
    print(cleaned)
    
    validated = validate_dataset(cleaned, required_columns=['id', 'value'], min_rows=3)
    print(f"\nDataset validation: {validated}")
    
    filtered = remove_outliers(cleaned, 'value', method='iqr')
    print("\nDataset after outlier removal:")
    print(filtered)