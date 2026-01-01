
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
    
    return filtered_df.reset_index(drop=True)

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    stats = {
        'original_count': len(df),
        'cleaned_count': len(df),
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max()
    }
    
    return stats

def process_dataset(file_path, column_to_clean):
    """
    Main function to load and clean a dataset.
    
    Parameters:
    file_path (str): Path to CSV file
    column_to_clean (str): Column name to clean
    
    Returns:
    tuple: (cleaned_df, statistics)
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded dataset with {len(df)} rows")
        
        original_stats = calculate_summary_statistics(df, column_to_clean)
        print(f"Original statistics for '{column_to_clean}':")
        print(f"  Mean: {original_stats['mean']:.2f}")
        print(f"  Std: {original_stats['std']:.2f}")
        
        cleaned_df = remove_outliers_iqr(df, column_to_clean)
        removed_count = len(df) - len(cleaned_df)
        print(f"Removed {removed_count} outliers ({removed_count/len(df)*100:.1f}%)")
        
        cleaned_stats = calculate_summary_statistics(cleaned_df, column_to_clean)
        cleaned_stats['outliers_removed'] = removed_count
        
        return cleaned_df, cleaned_stats
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        return None, None
    except Exception as e:
        print(f"Error processing dataset: {str(e)}")
        return None, None

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'values': np.concatenate([
            np.random.normal(100, 15, 95),
            np.random.normal(300, 50, 5)  # Outliers
        ])
    }
    
    df = pd.DataFrame(sample_data)
    cleaned_df, stats = process_dataset('sample_data.csv', 'values')
    
    if cleaned_df is not None:
        print(f"\nCleaned dataset has {len(cleaned_df)} rows")
        print(f"New mean: {stats['mean']:.2f}")
        print(f"New std: {stats['std']:.2f}")