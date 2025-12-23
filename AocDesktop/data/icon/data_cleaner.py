
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Args:
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
    
    Args:
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
        'count': len(df[column])
    }
    
    return stats

def process_dataset(file_path, column_to_clean):
    """
    Main function to load, clean, and analyze a dataset.
    
    Args:
        file_path (str): Path to CSV file
        column_to_clean (str): Column name to clean
    
    Returns:
        tuple: Cleaned DataFrame and statistics dictionary
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Original dataset shape: {df.shape}")
        
        cleaned_df = remove_outliers_iqr(df, column_to_clean)
        print(f"Cleaned dataset shape: {cleaned_df.shape}")
        print(f"Removed {len(df) - len(cleaned_df)} outliers")
        
        stats = calculate_summary_statistics(cleaned_df, column_to_clean)
        
        return cleaned_df, stats
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        return None, None
    except Exception as e:
        print(f"Error processing dataset: {str(e)}")
        return None, None
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        factor: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def zscore_normalize(data, column):
    """
    Normalize data using Z-score normalization.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        DataFrame with normalized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    data_copy = data.copy()
    mean_val = data_copy[column].mean()
    std_val = data_copy[column].std()
    
    if std_val == 0:
        data_copy[f'{column}_normalized'] = 0
    else:
        data_copy[f'{column}_normalized'] = (data_copy[column] - mean_val) / std_val
    
    return data_copy

def minmax_normalize(data, column, feature_range=(0, 1)):
    """
    Normalize data using Min-Max scaling.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
        feature_range: tuple of (min, max) for output range
    
    Returns:
        DataFrame with normalized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    data_copy = data.copy()
    min_val = data_copy[column].min()
    max_val = data_copy[column].max()
    
    if max_val == min_val:
        data_copy[f'{column}_normalized'] = feature_range[0]
    else:
        scaled = (data_copy[column] - min_val) / (max_val - min_val)
        data_copy[f'{column}_normalized'] = scaled * (feature_range[1] - feature_range[0]) + feature_range[0]
    
    return data_copy

def detect_skewed_columns(data, threshold=0.5):
    """
    Detect columns with skewed distributions.
    
    Args:
        data: pandas DataFrame
        threshold: absolute skewness threshold (default 0.5)
    
    Returns:
        List of column names with skewness above threshold
    """
    skewed_cols = []
    
    for column in data.select_dtypes(include=[np.number]).columns:
        skewness = data[column].skew()
        if abs(skewness) > threshold:
            skewed_cols.append((column, skewness))
    
    return sorted(skewed_cols, key=lambda x: abs(x[1]), reverse=True)

def log_transform(data, column):
    """
    Apply log transformation to reduce skewness.
    
    Args:
        data: pandas DataFrame
        column: column name to transform
    
    Returns:
        DataFrame with transformed column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    data_copy = data.copy()
    
    if (data_copy[column] <= 0).any():
        min_val = data_copy[column].min()
        if min_val <= 0:
            shift = abs(min_val) + 1
            data_copy[f'{column}_log'] = np.log(data_copy[column] + shift)
        else:
            data_copy[f'{column}_log'] = np.log(data_copy[column])
    else:
        data_copy[f'{column}_log'] = np.log(data_copy[column])
    
    return data_copy
import pandas as pd
import numpy as np
import sys

def clean_csv(input_file, output_file):
    try:
        df = pd.read_csv(input_file)
        
        print(f"Original shape: {df.shape}")
        print(f"Missing values per column:")
        print(df.isnull().sum())
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
        
        df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to {output_file}")
        print(f"Final shape: {df.shape}")
        
        return True
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
        return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python data_cleaner.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    success = clean_csv(input_file, output_file)
    sys.exit(0 if success else 1)