
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, columns):
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def normalize_data(df, columns, method='minmax'):
    df_norm = df.copy()
    for col in columns:
        if method == 'minmax':
            min_val = df_norm[col].min()
            max_val = df_norm[col].max()
            df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
        elif method == 'zscore':
            mean_val = df_norm[col].mean()
            std_val = df_norm[col].std()
            df_norm[col] = (df_norm[col] - mean_val) / std_val
    return df_norm

def clean_dataset(file_path, numeric_columns):
    df = pd.read_csv(file_path)
    df_clean = remove_outliers_iqr(df, numeric_columns)
    df_normalized = normalize_data(df_clean, numeric_columns, method='zscore')
    return df_normalized

if __name__ == "__main__":
    cleaned_data = clean_dataset('sample_data.csv', ['age', 'income', 'score'])
    cleaned_data.to_csv('cleaned_data.csv', index=False)
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Parameters:
    data (list or np.array): Input data
    column (int): Column index for 2D data, or None for 1D data
    
    Returns:
    np.array: Data with outliers removed
    """
    if column is not None:
        column_data = data[:, column]
    else:
        column_data = np.array(data)
    
    q1 = np.percentile(column_data, 25)
    q3 = np.percentile(column_data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    if column is not None:
        mask = (data[:, column] >= lower_bound) & (data[:, column] <= upper_bound)
        return data[mask]
    else:
        mask = (column_data >= lower_bound) & (column_data <= upper_bound)
        return column_data[mask]

def calculate_statistics(data):
    """
    Calculate basic statistics for the data.
    
    Parameters:
    data (np.array): Input data
    
    Returns:
    dict: Dictionary containing mean, median, std, min, max
    """
    return {
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data)
    }

def clean_dataset(data, columns_to_clean=None):
    """
    Clean dataset by removing outliers from specified columns.
    
    Parameters:
    data (np.array): 2D array of data
    columns_to_clean (list): List of column indices to clean
    
    Returns:
    np.array: Cleaned dataset
    """
    if columns_to_clean is None:
        columns_to_clean = range(data.shape[1])
    
    cleaned_data = data.copy()
    
    for col in columns_to_clean:
        if col < data.shape[1]:
            cleaned_data = remove_outliers_iqr(cleaned_data, col)
    
    return cleaned_data

if __name__ == "__main__":
    # Example usage
    sample_data = np.random.randn(100, 3) * 10 + 50
    print("Original data shape:", sample_data.shape)
    
    cleaned = clean_dataset(sample_data, [0, 1, 2])
    print("Cleaned data shape:", cleaned.shape)
    
    stats = calculate_statistics(cleaned[:, 0])
    print("Statistics for first column:", stats)import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from DataFrame.
    """
    if subset:
        return df.drop_duplicates(subset=subset, keep='first')
    else:
        return df.drop_duplicates(keep='first')

def convert_column_types(df, column_type_map):
    """
    Convert specified columns to given data types.
    """
    for column, dtype in column_type_map.items():
        if column in df.columns:
            df[column] = df[column].astype(dtype)
    return df

def handle_missing_values(df, strategy='drop', fill_value=None):
    """
    Handle missing values in DataFrame.
    """
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'fill':
        if fill_value is not None:
            return df.fillna(fill_value)
        else:
            return df.fillna(df.mean())
    else:
        return df

def normalize_column(df, column):
    """
    Normalize a column to range [0, 1].
    """
    if column in df.columns:
        col_min = df[column].min()
        col_max = df[column].max()
        if col_max != col_min:
            df[column] = (df[column] - col_min) / (col_max - col_min)
    return df

def clean_dataframe(df, operations):
    """
    Apply multiple cleaning operations to DataFrame.
    """
    for operation in operations:
        if operation['type'] == 'remove_duplicates':
            df = remove_duplicates(df, operation.get('subset'))
        elif operation['type'] == 'convert_types':
            df = convert_column_types(df, operation['column_type_map'])
        elif operation['type'] == 'handle_missing':
            df = handle_missing_values(df, 
                                      operation.get('strategy', 'drop'),
                                      operation.get('fill_value'))
        elif operation['type'] == 'normalize':
            df = normalize_column(df, operation['column'])
    return df

def validate_dataframe(df, rules):
    """
    Validate DataFrame against given rules.
    """
    violations = []
    for rule in rules:
        column = rule['column']
        rule_type = rule['type']
        
        if column not in df.columns:
            violations.append(f"Column '{column}' not found")
            continue
            
        if rule_type == 'not_null':
            null_count = df[column].isnull().sum()
            if null_count > 0:
                violations.append(f"Column '{column}' has {null_count} null values")
                
        elif rule_type == 'unique':
            duplicate_count = df[column].duplicated().sum()
            if duplicate_count > 0:
                violations.append(f"Column '{column}' has {duplicate_count} duplicate values")
                
        elif rule_type == 'range':
            min_val = rule.get('min')
            max_val = rule.get('max')
            if min_val is not None:
                below_min = (df[column] < min_val).sum()
                if below_min > 0:
                    violations.append(f"Column '{column}' has {below_min} values below {min_val}")
            if max_val is not None:
                above_max = (df[column] > max_val).sum()
                if above_max > 0:
                    violations.append(f"Column '{column}' has {above_max} values above {max_val}")
    
    return violations

def get_dataframe_stats(df):
    """
    Get basic statistics of DataFrame.
    """
    stats = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'unique_counts': {col: df[col].nunique() for col in df.columns}
    }
    return stats

def save_cleaned_data(df, filepath, format='csv'):
    """
    Save cleaned DataFrame to file.
    """
    if format == 'csv':
        df.to_csv(filepath, index=False)
    elif format == 'excel':
        df.to_excel(filepath, index=False)
    elif format == 'json':
        df.to_json(filepath, orient='records')
    else:
        raise ValueError(f"Unsupported format: {format}")