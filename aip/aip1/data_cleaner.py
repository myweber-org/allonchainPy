import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas Series using the IQR method.
    Returns a cleaned Series.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def normalize_minmax(data, column):
    """
    Normalize a column using min-max scaling.
    Returns a new Series with normalized values.
    """
    min_val = data[column].min()
    max_val = data[column].max()
    if max_val == min_val:
        return pd.Series([0.5] * len(data), index=data.index)
    return (data[column] - min_val) / (max_val - min_val)

def clean_dataset(df, numeric_columns):
    """
    Clean a dataset by removing outliers and normalizing numeric columns.
    Returns a cleaned DataFrame.
    """
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col + '_normalized'] = normalize_minmax(cleaned_df, col)
    return cleaned_df

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 200),
        'feature_b': np.random.exponential(50, 200),
        'category': np.random.choice(['X', 'Y', 'Z'], 200)
    })
    cleaned = clean_dataset(sample_data, ['feature_a', 'feature_b'])
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {cleaned.shape}")
    print(cleaned[['feature_a_normalized', 'feature_b_normalized']].head())import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def remove_outliers_zscore(data, column, threshold=3):
    z_scores = np.abs(stats.zscore(data[column]))
    return data[z_scores < threshold]

def normalize_minmax(data, column):
    min_val = data[column].min()
    max_val = data[column].max()
    data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    return data

def normalize_zscore(data, column):
    mean_val = data[column].mean()
    std_val = data[column].std()
    data[column + '_standardized'] = (data[column] - mean_val) / std_val
    return data

def clean_dataset(df, numeric_columns, method='iqr', normalize=True):
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        elif method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, col)
    
    if normalize:
        for col in numeric_columns:
            if col in cleaned_df.columns:
                cleaned_df = normalize_minmax(cleaned_df, col)
    
    return cleaned_df

def validate_data(df, required_columns, numeric_columns):
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    for col in numeric_columns:
        if col in df.columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
    
    return df
def remove_duplicates(data_list):
    """
    Remove duplicate entries from a list while preserving order.
    
    Args:
        data_list (list): Input list that may contain duplicates.
    
    Returns:
        list: List with duplicates removed.
    """
    seen = set()
    result = []
    for item in data_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def clean_numeric_data(values):
    """
    Clean numeric data by converting strings to floats and removing None values.
    
    Args:
        values (list): List of values that may contain strings, numbers, or None.
    
    Returns:
        list: Cleaned list of numeric values.
    """
    cleaned = []
    for val in values:
        if val is None:
            continue
        try:
            cleaned.append(float(val))
        except (ValueError, TypeError):
            continue
    return cleaned

if __name__ == "__main__":
    # Example usage
    sample_data = [1, 2, 2, 3, 4, 4, 5, 1]
    cleaned = remove_duplicates(sample_data)
    print(f"Original: {sample_data}")
    print(f"Cleaned: {cleaned}")
    
    mixed_data = ["10", 20, None, "30.5", "invalid", 40]
    numeric_data = clean_numeric_data(mixed_data)
    print(f"Numeric data: {numeric_data}")
import pandas as pd
import numpy as np
from scipy import stats

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    for column in df.select_dtypes(include=[np.number]).columns:
        df[column] = df[column].fillna(df[column].median())
    
    # Remove outliers using z-score
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    z_scores = np.abs(stats.zscore(df[numeric_cols]))
    df = df[(z_scores < 3).all(axis=1)]
    
    # Normalize numeric columns
    for column in numeric_cols:
        if df[column].std() != 0:
            df[column] = (df[column] - df[column].mean()) / df[column].std()
    
    return df

def save_cleaned_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = load_and_clean_data(input_file)
    save_cleaned_data(cleaned_df, output_file)import numpy as np
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
        return df[column].apply(lambda x: 0.0)
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

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 50),
        'feature_b': np.random.exponential(2.0, 50),
        'category': np.random.choice(['X', 'Y', 'Z'], 50)
    })
    
    numeric_cols = ['feature_a', 'feature_b']
    print("Original data shape:", sample_data.shape)
    
    try:
        validate_data(sample_data, numeric_cols)
        cleaned = clean_dataset(sample_data, numeric_cols)
        print("Cleaned data shape:", cleaned.shape)
        print("Cleaned data summary:")
        print(cleaned[numeric_cols].describe())
    except ValueError as e:
        print(f"Validation error: {e}")import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def clean_numeric_data(df, numeric_columns):
    """
    Clean numeric columns by removing outliers and filling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of numeric column names
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    
    return cleaned_df

def process_dataset(file_path, output_path=None):
    """
    Main function to process a dataset file.
    
    Parameters:
    file_path (str): Path to input CSV file
    output_path (str, optional): Path to save cleaned data
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    try:
        df = pd.read_csv(file_path)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        cleaned_df = clean_numeric_data(df, numeric_cols)
        
        if output_path:
            cleaned_df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
        
        print(f"Original shape: {df.shape}")
        print(f"Cleaned shape: {cleaned_df.shape}")
        print(f"Rows removed: {len(df) - len(cleaned_df)}")
        
        return cleaned_df
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'id': range(1, 101),
        'value': np.concatenate([
            np.random.normal(100, 10, 90),
            np.array([500, -100, 1000])
        ]),
        'score': np.concatenate([
            np.random.uniform(0, 100, 95),
            np.array([150, -50, 200, 300, -100])
        ])
    })
    
    cleaned = process_dataset(sample_data, 'cleaned_data.csv')
    
    if cleaned is not None:
        print("\nSummary statistics:")
        print(cleaned.describe())
import pandas as pd
import numpy as np

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

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a DataFrame column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    If no columns specified, clean all numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    cleaned_df = df.copy()
    
    for column in columns:
        if column in cleaned_df.columns:
            if pd.api.types.is_numeric_dtype(cleaned_df[column]):
                cleaned_df = remove_outliers_iqr(cleaned_df, column)
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 3, 4, 5, 100],
        'B': [10, 20, 30, 40, 50, 200],
        'C': ['a', 'b', 'c', 'd', 'e', 'f']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    cleaned = clean_numeric_data(df, ['A', 'B'])
    print("Cleaned DataFrame:")
    print(cleaned)
    print()
    
    stats = calculate_summary_statistics(df, 'A')
    print("Summary statistics for column 'A':")
    for key, value in stats.items():
        print(f"{key}: {value}")