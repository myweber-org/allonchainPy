
import pandas as pd
import numpy as np

def remove_outliers(df, column, threshold=3):
    mean = df[column].mean()
    std = df[column].std()
    z_scores = np.abs((df[column] - mean) / std)
    return df[z_scores < threshold]

def normalize_column(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(file_path, output_path):
    df = pd.read_csv(file_path)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        df = remove_outliers(df, col)
        df = normalize_column(df, col)
    
    df.to_csv(output_path, index=False)
    return df

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    cleaned_df = clean_dataset(input_file, output_file)
    print(f"Data cleaning completed. Cleaned data saved to {output_file}")
    print(f"Original shape: {pd.read_csv(input_file).shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a dataset using the Interquartile Range method.
    
    Args:
        data: numpy array or list of numerical values
        column: column index or key to process
        
    Returns:
        Cleaned data with outliers removed
    """
    if isinstance(data, list):
        data = np.array(data)
    
    q1 = np.percentile(data[:, column], 25)
    q3 = np.percentile(data[:, column], 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    mask = (data[:, column] >= lower_bound) & (data[:, column] <= upper_bound)
    cleaned_data = data[mask]
    
    return cleaned_data

def calculate_statistics(data, column):
    """
    Calculate basic statistics for a data column.
    
    Args:
        data: numpy array
        column: column index to analyze
        
    Returns:
        Dictionary containing mean, median, std, min, max
    """
    column_data = data[:, column]
    
    stats = {
        'mean': np.mean(column_data),
        'median': np.median(column_data),
        'std': np.std(column_data),
        'min': np.min(column_data),
        'max': np.max(column_data),
        'count': len(column_data)
    }
    
    return stats

def normalize_data(data, column, method='minmax'):
    """
    Normalize data in a column using specified method.
    
    Args:
        data: numpy array
        column: column index to normalize
        method: normalization method ('minmax' or 'zscore')
        
    Returns:
        Data with normalized column
    """
    column_data = data[:, column].astype(float)
    
    if method == 'minmax':
        min_val = np.min(column_data)
        max_val = np.max(column_data)
        if max_val - min_val != 0:
            normalized = (column_data - min_val) / (max_val - min_val)
        else:
            normalized = column_data
    elif method == 'zscore':
        mean_val = np.mean(column_data)
        std_val = np.std(column_data)
        if std_val != 0:
            normalized = (column_data - mean_val) / std_val
        else:
            normalized = column_data
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")
    
    data[:, column] = normalized
    return data

def validate_data(data, column, expected_type=float):
    """
    Validate data in a column meets expected type.
    
    Args:
        data: numpy array
        column: column index to validate
        expected_type: expected data type
        
    Returns:
        Boolean indicating if validation passed
    """
    try:
        column_data = data[:, column]
        converted = column_data.astype(expected_type)
        return True
    except (ValueError, TypeError):
        return False

if __name__ == "__main__":
    # Example usage
    sample_data = np.array([
        [1.0, 100.5, 10.2],
        [2.0, 102.3, 11.1],
        [3.0, 99.8, 9.8],
        [4.0, 150.0, 12.5],  # Potential outlier
        [5.0, 98.7, 10.0],
        [6.0, 101.2, 10.8]
    ])
    
    print("Original data shape:", sample_data.shape)
    
    # Remove outliers from column 1
    cleaned = remove_outliers_iqr(sample_data, 1)
    print("Cleaned data shape:", cleaned.shape)
    
    # Calculate statistics
    stats = calculate_statistics(cleaned, 1)
    print("Statistics:", stats)
    
    # Normalize data
    normalized = normalize_data(cleaned.copy(), 1, method='zscore')
    print("Normalized column 1:", normalized[:, 1])