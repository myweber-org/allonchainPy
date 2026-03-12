
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import pandas as pd
import numpy as np

def clean_dataset(df, column_mapping=None, drop_duplicates=True, fill_missing='median'):
    """
    Clean a pandas DataFrame by handling duplicates, missing values, and column renaming.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    column_mapping (dict): Dictionary mapping old column names to new ones
    drop_duplicates (bool): Whether to drop duplicate rows
    fill_missing (str): Strategy to fill missing values ('median', 'mean', 'mode', or 'drop')
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    
    df_clean = df.copy()
    
    if column_mapping:
        df_clean = df_clean.rename(columns=column_mapping)
    
    if drop_duplicates:
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed = initial_rows - len(df_clean)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing == 'drop':
        df_clean = df_clean.dropna()
        print("Dropped rows with missing values")
    elif fill_missing in ['median', 'mean', 'mode']:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df_clean[col].isnull().any():
                if fill_missing == 'median':
                    fill_value = df_clean[col].median()
                elif fill_missing == 'mean':
                    fill_value = df_clean[col].mean()
                elif fill_missing == 'mode':
                    fill_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 0
                
                df_clean[col] = df_clean[col].fillna(fill_value)
                print(f"Filled missing values in '{col}' with {fill_missing}: {fill_value}")
    
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna('Unknown')
            print(f"Filled missing categorical values in '{col}' with 'Unknown'")
    
    df_clean = df_clean.reset_index(drop=True)
    
    print(f"Cleaning complete. Final dataset shape: {df_clean.shape}")
    print(f"Original shape: {df.shape}")
    
    return df_clean

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the cleaned dataset meets basic requirements.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of column names that must be present
    min_rows (int): Minimum number of rows required
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    
    if len(df) < min_rows:
        print(f"Validation failed: Dataset has only {len(df)} rows, minimum required is {min_rows}")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Validation failed: Missing required columns: {missing_cols}")
            return False
    
    if df.isnull().any().any():
        null_counts = df.isnull().sum()
        null_cols = null_counts[null_counts > 0]
        print(f"Validation failed: Dataset still contains null values in columns: {list(null_cols.index)}")
        return False
    
    print("Data validation passed")
    return True

def save_cleaned_data(df, output_path, format='csv'):
    """
    Save the cleaned DataFrame to a file.
    
    Parameters:
    df (pd.DataFrame): Cleaned DataFrame to save
    output_path (str): Path to save the file
    format (str): File format ('csv', 'parquet', or 'json')
    """
    
    if format == 'csv':
        df.to_csv(output_path, index=False)
    elif format == 'parquet':
        df.to_parquet(output_path, index=False)
    elif format == 'json':
        df.to_json(output_path, orient='records', indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Cleaned data saved to {output_path} in {format} format")

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 3, 4, 5],
        'value': [10.5, 20.3, np.nan, 30.1, 40.0, np.nan],
        'category': ['A', 'B', 'C', 'C', None, 'A'],
        'score': [85, 92, 78, 78, 88, 91]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned_df = clean_dataset(
        df, 
        column_mapping={'id': 'identifier', 'value': 'measurement'},
        drop_duplicates=True,
        fill_missing='median'
    )
    
    print("\nCleaned dataset:")
    print(cleaned_df)
    
    is_valid = validate_data(cleaned_df, required_columns=['identifier', 'measurement', 'category', 'score'])
    
    if is_valid:
        save_cleaned_data(cleaned_df, 'cleaned_data.csv', format='csv')
import numpy as np

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data
import pandas as pd
import numpy as np
from scipy import stats

def load_data(filepath):
    return pd.read_csv(filepath)

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def remove_outliers_zscore(df, column, threshold=3):
    z_scores = np.abs(stats.zscore(df[column]))
    return df[z_scores < threshold]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def normalize_zscore(df, column):
    mean_val = df[column].mean()
    std_val = df[column].std()
    df[column + '_standardized'] = (df[column] - mean_val) / std_val
    return df

def clean_dataset(filepath, numeric_columns, outlier_method='iqr', normalize_method='minmax'):
    df = load_data(filepath)
    
    for col in numeric_columns:
        if outlier_method == 'iqr':
            df = remove_outliers_iqr(df, col)
        elif outlier_method == 'zscore':
            df = remove_outliers_zscore(df, col)
        
        if normalize_method == 'minmax':
            df = normalize_minmax(df, col)
        elif normalize_method == 'zscore':
            df = normalize_zscore(df, col)
    
    return df

if __name__ == "__main__":
    cleaned_data = clean_dataset(
        'sample_data.csv',
        ['age', 'income', 'score'],
        outlier_method='zscore',
        normalize_method='zscore'
    )
    print(f"Cleaned dataset shape: {cleaned_data.shape}")
    print(cleaned_data.head())