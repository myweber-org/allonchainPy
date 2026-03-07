
import pandas as pd
import numpy as np

def clean_dataframe(df, drop_duplicates=True, fill_missing='mean'):
    """
    Cleans a pandas DataFrame by removing duplicates and handling missing values.
    """
    cleaned_df = df.copy()

    if drop_duplicates:
        initial_rows = cleaned_df.shape[0]
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - cleaned_df.shape[0]
        print(f"Removed {removed} duplicate row(s).")

    if cleaned_df.isnull().sum().any():
        print("Handling missing values...")
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        non_numeric_cols = cleaned_df.select_dtypes(exclude=[np.number]).columns

        if fill_missing == 'mean' and len(numeric_cols) > 0:
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].mean())
        elif fill_missing == 'median' and len(numeric_cols) > 0:
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].median())
        elif fill_missing == 'zero' and len(numeric_cols) > 0:
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(0)

        for col in non_numeric_cols:
            if cleaned_df[col].isnull().any():
                cleaned_df[col] = cleaned_df[col].fillna('Unknown')

    print(f"Data cleaning complete. Final shape: {cleaned_df.shape}")
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, None, 6],
        'B': [5.1, None, 5.1, 8.2, 9.0, 8.2],
        'C': ['x', 'y', 'x', None, 'z', 'y']
    }
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaning DataFrame...")
    result_df = clean_dataframe(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(result_df)
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    removed_count = len(data) - len(filtered_data)
    
    return filtered_data, removed_count

def normalize_minmax(data, column):
    """
    Normalize data using min-max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using z-score normalization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(df, numeric_columns=None, outlier_factor=1.5):
    """
    Comprehensive data cleaning pipeline
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    removal_stats = {}
    
    for col in numeric_columns:
        if col in df.columns:
            cleaned_df, removed = remove_outliers_iqr(cleaned_df, col, outlier_factor)
            removal_stats[col] = removed
            
            cleaned_df[f"{col}_normalized"] = normalize_minmax(cleaned_df, col)
            cleaned_df[f"{col}_standardized"] = standardize_zscore(cleaned_df, col)
    
    return cleaned_df, removal_stats

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    null_counts = df.isnull().sum()
    duplicate_count = df.duplicated().sum()
    
    validation_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'null_counts': null_counts.to_dict(),
        'duplicate_rows': duplicate_count,
        'dtypes': df.dtypes.to_dict()
    }
    
    return validation_report

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    cleaned_data, stats = clean_dataset(sample_data)
    report = validate_dataframe(cleaned_data)
    
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {cleaned_data.shape}")
    print(f"Outliers removed: {stats}")
    print(f"Validation report: {report}")