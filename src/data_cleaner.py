
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Args:
        data: numpy array or list of numerical values
        column: index of the column to clean (if data is 2D)
    
    Returns:
        Cleaned data with outliers removed
    """
    if isinstance(data, list):
        data = np.array(data)
    
    # Handle 2D data (multiple columns)
    if data.ndim == 2:
        column_data = data[:, column]
    else:
        column_data = data
    
    # Calculate Q1, Q3 and IQR
    Q1 = np.percentile(column_data, 25)
    Q3 = np.percentile(column_data, 75)
    IQR = Q3 - Q1
    
    # Define outlier bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter data
    if data.ndim == 2:
        mask = (column_data >= lower_bound) & (column_data <= upper_bound)
        cleaned_data = data[mask]
    else:
        cleaned_data = column_data[(column_data >= lower_bound) & (column_data <= upper_bound)]
    
    return cleaned_data

def calculate_statistics(data):
    """
    Calculate basic statistics for the data.
    
    Args:
        data: numpy array of numerical values
    
    Returns:
        Dictionary containing mean, median, std, min and max
    """
    stats = {
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data)
    }
    return stats

def normalize_data(data):
    """
    Normalize data to range [0, 1] using min-max scaling.
    
    Args:
        data: numpy array of numerical values
    
    Returns:
        Normalized data
    """
    data_min = np.min(data)
    data_max = np.max(data)
    
    if data_max - data_min == 0:
        return np.zeros_like(data)
    
    normalized = (data - data_min) / (data_max - data_min)
    return normalized

if __name__ == "__main__":
    # Example usage
    sample_data = np.random.randn(1000) * 10 + 50  # Generate sample data
    
    print("Original data statistics:")
    original_stats = calculate_statistics(sample_data)
    for key, value in original_stats.items():
        print(f"{key}: {value:.4f}")
    
    cleaned_data = remove_outliers_iqr(sample_data, 0)
    
    print("\nCleaned data statistics:")
    cleaned_stats = calculate_statistics(cleaned_data)
    for key, value in cleaned_stats.items():
        print(f"{key}: {value:.4f}")
    
    print(f"\nRemoved {len(sample_data) - len(cleaned_data)} outliers")
    
    normalized_data = normalize_data(cleaned_data)
    print(f"\nNormalized data range: [{np.min(normalized_data):.4f}, {np.max(normalized_data):.4f}]")