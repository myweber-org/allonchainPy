
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Args:
        data (np.ndarray): Input data array
        column (int): Index of column to clean
    
    Returns:
        np.ndarray: Data with outliers removed
        np.ndarray: Boolean mask of kept values
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    
    if column >= data.shape[1]:
        raise ValueError("Column index out of bounds")
    
    column_data = data[:, column]
    
    q1 = np.percentile(column_data, 25)
    q3 = np.percentile(column_data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    mask = (column_data >= lower_bound) & (column_data <= upper_bound)
    
    cleaned_data = data[mask]
    
    return cleaned_data, mask

def calculate_statistics(data, column):
    """
    Calculate basic statistics for a data column.
    
    Args:
        data (np.ndarray): Input data array
        column (int): Index of column to analyze
    
    Returns:
        dict: Dictionary containing statistics
    """
    column_data = data[:, column]
    
    stats = {
        'mean': np.mean(column_data),
        'median': np.median(column_data),
        'std': np.std(column_data),
        'min': np.min(column_data),
        'max': np.max(column_data),
        'q1': np.percentile(column_data, 25),
        'q3': np.percentile(column_data, 75)
    }
    
    return stats

def clean_dataset(data, columns_to_clean=None):
    """
    Clean multiple columns in a dataset.
    
    Args:
        data (np.ndarray): Input data array
        columns_to_clean (list): List of column indices to clean
    
    Returns:
        np.ndarray: Cleaned dataset
        dict: Cleaning report for each column
    """
    if columns_to_clean is None:
        columns_to_clean = list(range(data.shape[1]))
    
    cleaned_data = data.copy()
    report = {}
    
    for col in columns_to_clean:
        original_count = len(cleaned_data)
        cleaned_data, mask = remove_outliers_iqr(cleaned_data, col)
        removed_count = original_count - len(cleaned_data)
        
        report[col] = {
            'original_samples': original_count,
            'removed_outliers': removed_count,
            'remaining_samples': len(cleaned_data),
            'removal_percentage': (removed_count / original_count) * 100
        }
    
    return cleaned_data, report

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    sample_data = np.random.randn(100, 3)
    sample_data[0, 0] = 100  # Add an outlier
    
    print("Original data shape:", sample_data.shape)
    
    cleaned, report = clean_dataset(sample_data, columns_to_clean=[0])
    
    print("Cleaned data shape:", cleaned.shape)
    print("Cleaning report:", report)