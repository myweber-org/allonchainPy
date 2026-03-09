
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    return filtered_df

def normalize_minmax(dataframe, columns):
    """
    Normalize specified columns using min-max scaling
    """
    df_normalized = dataframe.copy()
    for col in columns:
        if col in dataframe.columns:
            min_val = dataframe[col].min()
            max_val = dataframe[col].max()
            if max_val != min_val:
                df_normalized[col] = (dataframe[col] - min_val) / (max_val - min_val)
    return df_normalized

def standardize_zscore(dataframe, columns):
    """
    Standardize specified columns using z-score normalization
    """
    df_standardized = dataframe.copy()
    for col in columns:
        if col in dataframe.columns:
            mean_val = dataframe[col].mean()
            std_val = dataframe[col].std()
            if std_val > 0:
                df_standardized[col] = (dataframe[col] - mean_val) / std_val
    return df_standardized

def handle_missing_values(dataframe, strategy='mean', columns=None):
    """
    Handle missing values using specified strategy
    """
    df_processed = dataframe.copy()
    if columns is None:
        columns = dataframe.columns
    
    for col in columns:
        if col in dataframe.columns and dataframe[col].isnull().any():
            if strategy == 'mean':
                fill_value = dataframe[col].mean()
            elif strategy == 'median':
                fill_value = dataframe[col].median()
            elif strategy == 'mode':
                fill_value = dataframe[col].mode()[0]
            elif strategy == 'constant':
                fill_value = 0
            else:
                continue
                
            df_processed[col] = dataframe[col].fillna(fill_value)
    
    return df_processed

def get_data_summary(dataframe):
    """
    Generate comprehensive data summary statistics
    """
    summary = {
        'shape': dataframe.shape,
        'missing_values': dataframe.isnull().sum().to_dict(),
        'data_types': dataframe.dtypes.to_dict(),
        'numeric_stats': {}
    }
    
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        summary['numeric_stats'][col] = {
            'mean': dataframe[col].mean(),
            'std': dataframe[col].std(),
            'min': dataframe[col].min(),
            'max': dataframe[col].max(),
            'median': dataframe[col].median(),
            'skewness': dataframe[col].skew()
        }
    
    return summary
import numpy as np
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

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize_method='minmax'):
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if outlier_method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        elif outlier_method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, col)
        
        if normalize_method == 'minmax':
            cleaned_df = normalize_minmax(cleaned_df, col)
        elif normalize_method == 'zscore':
            cleaned_df = normalize_zscore(cleaned_df, col)
    
    return cleaned_df

def get_summary_statistics(df, numeric_columns):
    summary = {}
    for col in numeric_columns:
        summary[col] = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'count': df[col].count(),
            'missing': df[col].isnull().sum()
        }
    return pd.DataFrame(summary).Timport pandas as pd
import numpy as np

def clean_dataset(df, strategy='mean', outlier_threshold=3):
    """
    Clean a pandas DataFrame by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    strategy (str): Strategy for missing value imputation ('mean', 'median', 'mode', 'drop').
    outlier_threshold (float): Number of standard deviations to consider as outlier.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if strategy == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif strategy in ['mean', 'median', 'mode']:
        for column in cleaned_df.select_dtypes(include=[np.number]).columns:
            if cleaned_df[column].isnull().any():
                if strategy == 'mean':
                    fill_value = cleaned_df[column].mean()
                elif strategy == 'median':
                    fill_value = cleaned_df[column].median()
                elif strategy == 'mode':
                    fill_value = cleaned_df[column].mode()[0]
                cleaned_df[column] = cleaned_df[column].fillna(fill_value)
    
    # Handle outliers for numerical columns
    numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        mean = cleaned_df[column].mean()
        std = cleaned_df[column].std()
        lower_bound = mean - outlier_threshold * std
        upper_bound = mean + outlier_threshold * std
        
        # Cap outliers to bounds
        cleaned_df[column] = cleaned_df[column].clip(lower=lower_bound, upper=upper_bound)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"

# Example usage
if __name__ == "__main__":
    # Create sample data with missing values and outliers
    sample_data = {
        'A': [1, 2, np.nan, 4, 100],  # Contains NaN and outlier
        'B': [5, 6, 7, np.nan, 9],
        'C': ['x', 'y', 'z', 'x', 'y']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    # Clean the data
    cleaned = clean_dataset(df, strategy='mean', outlier_threshold=2)
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    # Validate the cleaned data
    is_valid, message = validate_dataframe(cleaned, required_columns=['A', 'B', 'C'])
    print(f"\nValidation: {message}")