
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def remove_outliers_zscore(df, column, threshold=3):
    z_scores = np.abs(stats.zscore(df[column]))
    return df[z_scores < threshold]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def normalize_zscore(df, column):
    mean_val = df[column].mean()
    std_val = df[column].std()
    df[column + '_standardized'] = (df[column] - mean_val) / std_val
    return df

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize_method='minmax'):
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if outlier_method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        elif outlier_method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, col)
    
    for col in numeric_columns:
        if normalize_method == 'minmax':
            cleaned_df = normalize_minmax(cleaned_df, col)
        elif normalize_method == 'zscore':
            cleaned_df = normalize_zscore(cleaned_df, col)
    
    return cleaned_df

def validate_data(df, required_columns):
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    numeric_check = df.select_dtypes(include=[np.number])
    if len(numeric_check.columns) == 0:
        raise ValueError("No numeric columns found in dataset")
    
    return True

if __name__ == "__main__":
    sample_data = {
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'feature3': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    print(f"Original dataset shape: {df.shape}")
    
    try:
        validate_data(df, ['feature1', 'feature2', 'feature3'])
        cleaned_df = clean_dataset(df, ['feature1', 'feature2', 'feature3'])
        print(f"Cleaned dataset shape: {cleaned_df.shape}")
        print("Data cleaning completed successfully")
    except ValueError as e:
        print(f"Validation error: {e}")
import pandas as pd
import numpy as np

def clean_dataset(df, columns_to_check=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    # Create a copy to avoid modifying the original DataFrame
    df_clean = df.copy()
    
    # Remove duplicate rows
    initial_rows = df_clean.shape[0]
    df_clean = df_clean.drop_duplicates()
    removed_duplicates = initial_rows - df_clean.shape[0]
    
    # Handle missing values
    if columns_to_check is None:
        columns_to_check = df_clean.columns
    
    missing_counts = {}
    for column in columns_to_check:
        if column in df_clean.columns:
            missing_count = df_clean[column].isnull().sum()
            missing_counts[column] = missing_count
            
            # Fill numeric columns with median, categorical with mode
            if pd.api.types.is_numeric_dtype(df_clean[column]):
                median_value = df_clean[column].median()
                df_clean[column] = df_clean[column].fillna(median_value)
            else:
                mode_value = df_clean[column].mode()[0] if not df_clean[column].mode().empty else 'Unknown'
                df_clean[column] = df_clean[column].fillna(mode_value)
    
    # Remove outliers using IQR method for numeric columns
    numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
    outliers_removed = 0
    
    for column in numeric_columns:
        Q1 = df_clean[column].quantile(0.25)
        Q3 = df_clean[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        before_count = df_clean.shape[0]
        df_clean = df_clean[(df_clean[column] >= lower_bound) & (df_clean[column] <= upper_bound)]
        outliers_removed += (before_count - df_clean.shape[0])
    
    # Reset index after cleaning
    df_clean = df_clean.reset_index(drop=True)
    
    # Print cleaning summary
    print(f"Cleaning Summary:")
    print(f"  - Removed {removed_duplicates} duplicate rows")
    print(f"  - Handled missing values in {len(missing_counts)} columns")
    print(f"  - Removed {outliers_removed} outlier rows")
    print(f"  - Final dataset shape: {df_clean.shape}")
    
    return df_clean

def validate_data(df, required_columns=None):
    """
    Validate the DataFrame for required columns and data types.
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for any remaining missing values
    if df.isnull().any().any():
        print("Warning: Dataset still contains missing values")
    
    return True

# Example usage
if __name__ == "__main__":
    # Create sample data for demonstration
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5, 6],
        'name': ['Alice', 'Bob', 'Charlie', None, 'Eve', 'Eve', 'Frank'],
        'age': [25, 30, 35, 40, 200, 45, 50],  # 200 is an outlier
        'score': [85.5, 90.0, None, 75.5, 88.0, 88.0, 92.5],
        'category': ['A', 'B', 'A', None, 'C', 'C', 'B']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the dataset
    cleaned_df = clean_dataset(df)
    
    print("\nCleaned dataset:")
    print(cleaned_df)
    
    # Validate the cleaned data
    try:
        validate_data(cleaned_df, required_columns=['id', 'name', 'age', 'score'])
        print("\nData validation passed!")
    except ValueError as e:
        print(f"\nData validation failed: {e}")