def deduplicate_list(input_list):
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        multiplier: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return pd.Series([0.5] * len(data), index=data.index)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using Z-score normalization.
    
    Args:
        data: pandas DataFrame
        column: column name to standardize
    
    Returns:
        Series with standardized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(data, numeric_columns=None, outlier_multiplier=1.5):
    """
    Clean dataset by removing outliers and normalizing numeric columns.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric column names to process
        outlier_multiplier: IQR multiplier for outlier detection
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column in data.columns:
            # Remove outliers
            q1 = data[column].quantile(0.25)
            q3 = data[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - outlier_multiplier * iqr
            upper_bound = q3 + outlier_multiplier * iqr
            
            mask = (cleaned_data[column] >= lower_bound) & (cleaned_data[column] <= upper_bound)
            cleaned_data = cleaned_data[mask]
            
            # Normalize the column
            cleaned_data[column] = normalize_minmax(cleaned_data, column)
    
    return cleaned_data.reset_index(drop=True)

def validate_data(data, required_columns, numeric_columns=None):
    """
    Validate dataset structure and content.
    
    Args:
        data: pandas DataFrame to validate
        required_columns: list of required column names
        numeric_columns: list of columns that should be numeric
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        return False, f"Missing required columns: {missing_columns}"
    
    if numeric_columns:
        non_numeric = [col for col in numeric_columns if col in data.columns 
                      and not pd.api.types.is_numeric_dtype(data[col])]
        if non_numeric:
            return False, f"Columns should be numeric: {non_numeric}"
    
    if data.empty:
        return False, "Dataset is empty"
    
    return True, "Dataset is valid"

# Example usage function
def process_sample_data():
    """
    Demonstrate the data cleaning functions with sample data.
    """
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'id': range(100),
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.exponential(50, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })
    
    print("Original data shape:", sample_data.shape)
    print("Original statistics:")
    print(sample_data[['feature_a', 'feature_b']].describe())
    
    # Remove outliers from feature_a
    cleaned = remove_outliers_iqr(sample_data, 'feature_a')
    print("\nAfter outlier removal shape:", cleaned.shape)
    
    # Normalize feature_b
    normalized = normalize_minmax(cleaned, 'feature_b')
    print("\nNormalized feature_b statistics:")
    print(normalized.describe())
    
    # Full cleaning
    fully_cleaned = clean_dataset(sample_data, ['feature_a', 'feature_b'])
    print("\nFully cleaned data shape:", fully_cleaned.shape)
    
    # Validate
    is_valid, message = validate_data(fully_cleaned, ['feature_a', 'feature_b'], ['feature_a', 'feature_b'])
    print(f"\nValidation: {message}")
    
    return fully_cleaned

if __name__ == "__main__":
    cleaned_data = process_sample_data()