import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Parameters:
    data (list or array-like): The dataset
    column (int or str): Column index or name if using pandas DataFrame
    
    Returns:
    numpy.ndarray: Data with outliers removed
    """
    if isinstance(data, list):
        data = np.array(data)
    
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
    
    return filtered_data

def calculate_basic_stats(data):
    """
    Calculate basic statistics for the data.
    
    Parameters:
    data (array-like): Input data
    
    Returns:
    dict: Dictionary containing mean, median, std, min, max
    """
    if isinstance(data, list):
        data = np.array(data)
    
    stats = {
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data)
    }
    
    return stats

def clean_dataset(data, columns=None):
    """
    Clean dataset by removing outliers from specified columns.
    
    Parameters:
    data (array-like): Input dataset
    columns (list): List of column indices to clean
    
    Returns:
    numpy.ndarray: Cleaned dataset
    """
    if columns is None:
        columns = range(data.shape[1])
    
    cleaned_data = data.copy()
    
    for col in columns:
        col_data = data[:, col]
        cleaned_col = remove_outliers_iqr(col_data, col)
        
        if len(cleaned_col) < len(col_data):
            print(f"Removed {len(col_data) - len(cleaned_col)} outliers from column {col}")
        
        cleaned_data = cleaned_data[cleaned_data[:, col] >= np.min(cleaned_col)]
        cleaned_data = cleaned_data[cleaned_data[:, col] <= np.max(cleaned_col)]
    
    return cleaned_data

if __name__ == "__main__":
    sample_data = np.random.randn(100, 3)
    sample_data[0, 0] = 10
    sample_data[1, 1] = -8
    
    print("Original data shape:", sample_data.shape)
    print("Sample statistics:", calculate_basic_stats(sample_data[:, 0]))
    
    cleaned = clean_dataset(sample_data)
    print("Cleaned data shape:", cleaned.shape)
    print("Cleaned statistics:", calculate_basic_stats(cleaned[:, 0]))