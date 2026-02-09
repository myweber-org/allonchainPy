
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def clean_dataset(df, numeric_columns):
    original_shape = df.shape
    for col in numeric_columns:
        if col in df.columns:
            df = remove_outliers_iqr(df, col)
    cleaned_shape = df.shape
    removed_count = original_shape[0] - cleaned_shape[0]
    print(f"Removed {removed_count} outliers from dataset")
    return df

def main():
    sample_data = {
        'temperature': [22, 23, 24, 25, 26, 100, 27, 28, 29, -10],
        'humidity': [45, 46, 47, 48, 49, 200, 50, 51, 52, -5],
        'pressure': [1013, 1014, 1015, 1016, 1017, 2000, 1018, 1019, 1020, 500]
    }
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    
    numeric_cols = ['temperature', 'humidity', 'pressure']
    cleaned_df = clean_dataset(df, numeric_cols)
    
    print("\nCleaned dataset:")
    print(cleaned_df)
    
    return cleaned_df

if __name__ == "__main__":
    cleaned_data = main()import pandas as pd
import numpy as np

def clean_csv_data(filepath, fill_strategy='mean', drop_threshold=0.5):
    """
    Load and clean CSV data by handling missing values.
    
    Parameters:
    filepath (str): Path to the CSV file
    fill_strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', 'zero')
    drop_threshold (float): Drop columns with missing ratio above this threshold
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise ValueError(f"File not found: {filepath}")
    
    missing_ratio = df.isnull().sum() / len(df)
    columns_to_drop = missing_ratio[missing_ratio > drop_threshold].index
    df = df.drop(columns=columns_to_drop)
    
    for column in df.columns:
        if df[column].dtype in ['float64', 'int64']:
            if fill_strategy == 'mean':
                fill_value = df[column].mean()
            elif fill_strategy == 'median':
                fill_value = df[column].median()
            elif fill_strategy == 'zero':
                fill_value = 0
            else:
                fill_value = df[column].mode()[0] if not df[column].mode().empty else 0
            df[column] = df[column].fillna(fill_value)
        else:
            df[column] = df[column].fillna(df[column].mode()[0] if not df[column].mode().empty else 'Unknown')
    
    return df

def detect_outliers_iqr(df, column):
    """
    Detect outliers using IQR method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to check for outliers
    
    Returns:
    pd.Series: Boolean series indicating outliers
    """
    if df[column].dtype not in ['float64', 'int64']:
        return pd.Series([False] * len(df))
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    return (df[column] < lower_bound) | (df[column] > upper_bound)

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    subset (list): Columns to consider for duplication
    keep (str): Which duplicates to keep ('first', 'last', False)
    
    Returns:
    pd.DataFrame: DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def normalize_column(df, column):
    """
    Normalize column values to range [0, 1].
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to normalize
    
    Returns:
    pd.Series: Normalized column values
    """
    if df[column].dtype not in ['float64', 'int64']:
        return df[column]
    
    min_val = df[column].min()
    max_val = df[column].max()
    
    if max_val == min_val:
        return pd.Series([0.5] * len(df))
    
    return (df[column] - min_val) / (max_val - min_val)