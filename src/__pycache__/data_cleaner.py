
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def calculate_statistics(df, column):
    """
    Calculate basic statistics for a column after outlier removal.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing statistical measures
    """
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

def clean_dataset(df, numeric_columns):
    """
    Clean dataset by removing outliers from multiple numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    dict: Statistics for each cleaned column
    """
    cleaned_df = df.copy()
    statistics = {}
    
    for col in numeric_columns:
        if col in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[col]):
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            removed_count = original_count - len(cleaned_df)
            
            stats = calculate_statistics(cleaned_df, col)
            stats['outliers_removed'] = removed_count
            statistics[col] = stats
    
    return cleaned_df, statistics

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Create sample data with outliers
    data = {
        'temperature': np.random.normal(25, 5, 1000),
        'humidity': np.random.normal(60, 15, 1000),
        'pressure': np.random.normal(1013, 10, 1000)
    }
    
    # Add some outliers
    data['temperature'][:50] = np.random.uniform(50, 100, 50)
    data['humidity'][:30] = np.random.uniform(0, 10, 30)
    
    df = pd.DataFrame(data)
    
    print("Original dataset shape:", df.shape)
    print("\nOriginal statistics:")
    for col in df.columns:
        print(f"{col}: mean={df[col].mean():.2f}, std={df[col].std():.2f}")
    
    # Clean the dataset
    numeric_cols = ['temperature', 'humidity', 'pressure']
    cleaned_df, stats = clean_dataset(df, numeric_cols)
    
    print("\nCleaned dataset shape:", cleaned_df.shape)
    print("\nCleaned statistics:")
    for col, col_stats in stats.items():
        print(f"\n{col}:")
        for stat_name, value in col_stats.items():
            print(f"  {stat_name}: {value:.2f}" if isinstance(value, float) else f"  {stat_name}: {value}")import pandas as pd

def clean_data(df):
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Fill missing numeric values with column mean
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # Fill missing categorical values with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # Remove rows where critical columns are still null
    critical_columns = ['id', 'timestamp']
    if all(col in df.columns for col in critical_columns):
        df = df.dropna(subset=critical_columns)
    
    return df

def validate_data(df):
    # Check for remaining null values
    if df.isnull().sum().sum() > 0:
        print("Warning: Data still contains null values")
        return False
    
    # Check for negative values in positive-only columns
    positive_columns = ['age', 'price', 'quantity']
    for col in positive_columns:
        if col in df.columns:
            if (df[col] < 0).any():
                print(f"Warning: Column {col} contains negative values")
                return False
    
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = pd.DataFrame({
        'id': [1, 2, 2, 3, None],
        'name': ['Alice', 'Bob', 'Bob', None, 'Eve'],
        'age': [25, 30, 30, None, -5],
        'score': [85.5, 92.0, 92.0, 78.5, None]
    })
    
    print("Original data:")
    print(sample_data)
    
    cleaned_data = clean_data(sample_data)
    print("\nCleaned data:")
    print(cleaned_data)
    
    is_valid = validate_data(cleaned_data)
    print(f"\nData validation passed: {is_valid}")