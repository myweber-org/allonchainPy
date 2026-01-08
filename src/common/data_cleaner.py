import pandas as pd
import numpy as np
from scipy import stats

def load_and_clean_data(filepath):
    """Load CSV data and perform cleaning operations."""
    df = pd.read_csv(filepath)
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Handle missing values by column mean
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # Remove outliers using z-score method
    z_scores = np.abs(stats.zscore(df[numeric_cols]))
    df = df[(z_scores < 3).all(axis=1)]
    
    # Normalize numeric columns
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
    
    return df

def save_cleaned_data(df, output_path):
    """Save cleaned DataFrame to CSV."""
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = load_and_clean_data(input_file)
    save_cleaned_data(cleaned_df, output_file)
    
    print(f"Data cleaning complete. Original shape: {pd.read_csv(input_file).shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")
    print(f"Saved to: {output_file}")
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, multiplier=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    
    return filtered_df

def normalize_minmax(dataframe, columns=None):
    """
    Normalize data using min-max scaling
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns
    
    normalized_df = dataframe.copy()
    
    for col in columns:
        if col in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe[col]):
            min_val = dataframe[col].min()
            max_val = dataframe[col].max()
            
            if max_val != min_val:
                normalized_df[col] = (dataframe[col] - min_val) / (max_val - min_val)
            else:
                normalized_df[col] = 0
    
    return normalized_df

def detect_missing_patterns(dataframe, threshold=0.3):
    """
    Detect columns with high percentage of missing values
    """
    missing_stats = {}
    
    for col in dataframe.columns:
        missing_count = dataframe[col].isnull().sum()
        missing_percentage = missing_count / len(dataframe)
        
        if missing_percentage > threshold:
            missing_stats[col] = {
                'missing_count': missing_count,
                'missing_percentage': missing_percentage,
                'action': 'consider_dropping'
            }
        elif missing_percentage > 0:
            missing_stats[col] = {
                'missing_count': missing_count,
                'missing_percentage': missing_percentage,
                'action': 'imputation_needed'
            }
    
    return missing_stats

def validate_data_types(dataframe, expected_types):
    """
    Validate column data types against expected types
    """
    validation_results = {}
    
    for col, expected_type in expected_types.items():
        if col not in dataframe.columns:
            validation_results[col] = {
                'status': 'missing',
                'actual': None,
                'expected': expected_type
            }
            continue
        
        actual_type = str(dataframe[col].dtype)
        
        if actual_type == expected_type:
            validation_results[col] = {
                'status': 'valid',
                'actual': actual_type,
                'expected': expected_type
            }
        else:
            validation_results[col] = {
                'status': 'invalid',
                'actual': actual_type,
                'expected': expected_type
            }
    
    return validation_results

def clean_dataset(dataframe, config=None):
    """
    Main cleaning function with configurable pipeline
    """
    if config is None:
        config = {
            'remove_outliers': True,
            'normalize': True,
            'outlier_multiplier': 1.5,
            'missing_threshold': 0.3
        }
    
    cleaned_df = dataframe.copy()
    
    if config.get('remove_outliers', False):
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            cleaned_df = remove_outliers_iqr(
                cleaned_df, 
                col, 
                multiplier=config.get('outlier_multiplier', 1.5)
            )
    
    if config.get('normalize', False):
        cleaned_df = normalize_minmax(cleaned_df)
    
    return cleaned_df

def generate_cleaning_report(dataframe):
    """
    Generate comprehensive data cleaning report
    """
    report = {
        'original_shape': dataframe.shape,
        'numeric_columns': list(dataframe.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(dataframe.select_dtypes(include=['object']).columns),
        'missing_patterns': detect_missing_patterns(dataframe),
        'summary_stats': dataframe.describe().to_dict()
    }
    
    return report