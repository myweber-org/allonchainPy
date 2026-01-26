
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, columns):
    df_clean = df.copy()
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def normalize_data(df, columns, method='minmax'):
    df_norm = df.copy()
    for col in columns:
        if col in df.columns:
            if method == 'minmax':
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val != min_val:
                    df_norm[col] = (df[col] - min_val) / (max_val - min_val)
                else:
                    df_norm[col] = 0
            elif method == 'zscore':
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val > 0:
                    df_norm[col] = (df[col] - mean_val) / std_val
                else:
                    df_norm[col] = 0
    return df_norm

def clean_dataset(df, numeric_columns):
    df_no_outliers = remove_outliers_iqr(df, numeric_columns)
    df_normalized = normalize_data(df_no_outliers, numeric_columns, method='zscore')
    df_final = df_normalized.dropna(subset=numeric_columns)
    return df_final

if __name__ == "__main__":
    sample_data = {
        'feature1': [1, 2, 3, 4, 100, 6, 7, 8, 9, 10],
        'feature2': [15, 20, 25, 30, 35, 40, 200, 50, 55, 60],
        'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
    }
    
    df = pd.DataFrame(sample_data)
    numeric_cols = ['feature1', 'feature2']
    
    cleaned_df = clean_dataset(df, numeric_cols)
    print("Original dataset shape:", df.shape)
    print("Cleaned dataset shape:", cleaned_df.shape)
    print("\nCleaned data summary:")
    print(cleaned_df.describe())
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
        removed_duplicates = initial_rows - len(cleaned_df)
        print(f"Removed {removed_duplicates} duplicate rows")
    
    # Handle missing values
    if fill_missing:
        for column in cleaned_df.columns:
            if cleaned_df[column].dtype in [np.float64, np.int64]:
                if cleaned_df[column].isnull().any():
                    if fill_strategy == 'mean':
                        fill_value = cleaned_df[column].mean()
                    elif fill_strategy == 'median':
                        fill_value = cleaned_df[column].median()
                    elif fill_strategy == 'mode':
                        fill_value = cleaned_df[column].mode()[0]
                    elif fill_strategy == 'zero':
                        fill_value = 0
                    else:
                        fill_value = cleaned_df[column].mean()
                    
                    cleaned_df[column] = cleaned_df[column].fillna(fill_value)
                    print(f"Filled missing values in column '{column}' with {fill_strategy}: {fill_value}")
            elif cleaned_df[column].dtype == 'object':
                if cleaned_df[column].isnull().any():
                    cleaned_df[column] = cleaned_df[column].fillna('Unknown')
                    print(f"Filled missing values in column '{column}' with 'Unknown'")
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate the dataset for basic integrity checks.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        dict: Dictionary with validation results
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        validation_results['missing_required_columns'] = missing_columns
    
    return validation_results

# Example usage
if __name__ == "__main__":
    # Create sample data with some issues
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', None, 'Eve'],
        'age': [25, 30, 30, None, 35, 28],
        'score': [85.5, 92.0, 92.0, 78.5, None, 88.0]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the dataset
    cleaned_df = clean_dataset(df, fill_strategy='median')
    print("\nCleaned dataset:")
    print(cleaned_df)
    
    # Validate the cleaned dataset
    validation = validate_dataset(cleaned_df, required_columns=['id', 'name', 'age'])
    print("\nValidation results:")
    for key, value in validation.items():
        print(f"{key}: {value}")