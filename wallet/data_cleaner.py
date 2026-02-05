
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas DataFrame column using the IQR method.
    
    Parameters:
    data (pd.DataFrame): The DataFrame containing the data.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed from the specified column.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def calculate_basic_stats(data, column):
    """
    Calculate basic statistics for a column after outlier removal.
    
    Parameters:
    data (pd.DataFrame): The DataFrame containing the data.
    column (str): The column name to analyze.
    
    Returns:
    dict: Dictionary containing mean, median, and standard deviation.
    """
    if data.empty:
        return {"mean": np.nan, "median": np.nan, "std": np.nan}
    
    stats = {
        "mean": data[column].mean(),
        "median": data[column].median(),
        "std": data[column].std()
    }
    return stats

if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'values': np.concatenate([
            np.random.normal(100, 10, 95),
            np.random.normal(200, 10, 5)  # Outliers
        ])
    })
    
    print("Original data shape:", sample_data.shape)
    print("Original stats:", calculate_basic_stats(sample_data, 'values'))
    
    # Remove outliers
    cleaned_data = remove_outliers_iqr(sample_data, 'values')
    
    print("\nCleaned data shape:", cleaned_data.shape)
    print("Cleaned stats:", calculate_basic_stats(cleaned_data, 'values'))