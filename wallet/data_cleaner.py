import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from DataFrame.
    """
    if subset is None:
        subset = df.columns.tolist()
    return df.drop_duplicates(subset=subset, keep='first')

def convert_column_types(df, column_type_map):
    """
    Convert specified columns to given data types.
    """
    for column, dtype in column_type_map.items():
        if column in df.columns:
            df[column] = df[column].astype(dtype)
    return df

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values using specified strategy.
    """
    if columns is None:
        columns = df.columns[df.isnull().any()].tolist()
    
    for column in columns:
        if strategy == 'mean' and pd.api.types.is_numeric_dtype(df[column]):
            df[column].fillna(df[column].mean(), inplace=True)
        elif strategy == 'median' and pd.api.types.is_numeric_dtype(df[column]):
            df[column].fillna(df[column].median(), inplace=True)
        elif strategy == 'mode':
            df[column].fillna(df[column].mode()[0], inplace=True)
        elif strategy == 'drop':
            df.dropna(subset=[column], inplace=True)
    
    return df

def normalize_column(df, column, method='minmax'):
    """
    Normalize specified column using given method.
    """
    if method == 'minmax':
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val != min_val:
            df[column] = (df[column] - min_val) / (max_val - min_val)
    elif method == 'zscore':
        mean_val = df[column].mean()
        std_val = df[column].std()
        if std_val != 0:
            df[column] = (df[column] - mean_val) / std_val
    
    return df

def clean_dataframe(df, operations):
    """
    Apply multiple cleaning operations to DataFrame.
    """
    for operation in operations:
        if operation['type'] == 'remove_duplicates':
            df = remove_duplicates(df, operation.get('subset'))
        elif operation['type'] == 'convert_types':
            df = convert_column_types(df, operation['type_map'])
        elif operation['type'] == 'handle_missing':
            df = handle_missing_values(
                df, 
                operation.get('strategy', 'mean'),
                operation.get('columns')
            )
        elif operation['type'] == 'normalize':
            df = normalize_column(
                df,
                operation['column'],
                operation.get('method', 'minmax')
            )
    
    return df
def remove_duplicates_preserve_order(seq):
    seen = set()
    result = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_missing (str): Strategy to fill missing values ('mean', 'median', 'mode', or 'drop').
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
            elif fill_missing == 'median':
                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            mode_value = cleaned_df[col].mode()
            if not mode_value.empty:
                cleaned_df[col].fillna(mode_value[0], inplace=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, None],
        'B': [5, None, 7, 8, 9],
        'C': ['x', 'y', 'y', 'z', 'z']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame (mean imputation):")
    cleaned = clean_dataset(df, fill_missing='mean')
    print(cleaned)
    
    is_valid, message = validate_dataframe(cleaned, required_columns=['A', 'B', 'C'])
    print(f"\nValidation: {message}")
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    removed_count = len(data) - len(filtered_data)
    
    return filtered_data, removed_count

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
    removed_count = len(data) - len(filtered_data)
    
    return filtered_data, removed_count

def normalize_minmax(data, columns=None):
    """
    Normalize data using Min-Max scaling.
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    normalized_data = data.copy()
    
    for col in columns:
        if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
            min_val = data[col].min()
            max_val = data[col].max()
            
            if max_val > min_val:
                normalized_data[col] = (data[col] - min_val) / (max_val - min_val)
    
    return normalized_data

def normalize_zscore(data, columns=None):
    """
    Normalize data using Z-score standardization.
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    standardized_data = data.copy()
    
    for col in columns:
        if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
            mean_val = data[col].mean()
            std_val = data[col].std()
            
            if std_val > 0:
                standardized_data[col] = (data[col] - mean_val) / std_val
    
    return standardized_data

def clean_dataset(data, outlier_method='iqr', outlier_columns=None, 
                  normalize_method='minmax', normalize_columns=None):
    """
    Comprehensive data cleaning pipeline.
    """
    cleaned_data = data.copy()
    
    if outlier_columns is None:
        outlier_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    total_removed = 0
    
    for col in outlier_columns:
        if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
            if outlier_method == 'iqr':
                cleaned_data, removed = remove_outliers_iqr(cleaned_data, col)
            elif outlier_method == 'zscore':
                cleaned_data, removed = remove_outliers_zscore(cleaned_data, col)
            else:
                raise ValueError("Invalid outlier_method. Use 'iqr' or 'zscore'")
            
            total_removed += removed
    
    if normalize_method == 'minmax':
        cleaned_data = normalize_minmax(cleaned_data, normalize_columns)
    elif normalize_method == 'zscore':
        cleaned_data = normalize_zscore(cleaned_data, normalize_columns)
    elif normalize_method is not None:
        raise ValueError("Invalid normalize_method. Use 'minmax', 'zscore', or None")
    
    return cleaned_data, total_removed

def get_data_summary(data):
    """
    Generate summary statistics for the dataset.
    """
    summary = {
        'total_rows': len(data),
        'total_columns': len(data.columns),
        'numeric_columns': data.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': data.select_dtypes(include=['object', 'category']).columns.tolist(),
        'missing_values': data.isnull().sum().sum(),
        'missing_percentage': (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
    }
    
    if len(summary['numeric_columns']) > 0:
        numeric_stats = data[summary['numeric_columns']].describe().to_dict()
        summary['numeric_statistics'] = numeric_stats
    
    return summary
import pandas as pd
import numpy as np
from scipy import stats

def load_dataset(filepath):
    return pd.read_csv(filepath)

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_column(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def handle_missing_values(df, strategy='mean'):
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            if strategy == 'mean':
                df[col].fillna(df[col].mean(), inplace=True)
            elif strategy == 'median':
                df[col].fillna(df[col].median(), inplace=True)
            elif strategy == 'mode':
                df[col].fillna(df[col].mode()[0], inplace=True)
    return df

def clean_dataframe(df, numeric_columns):
    df_clean = df.copy()
    df_clean = handle_missing_values(df_clean)
    
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean = remove_outliers_iqr(df_clean, col)
            df_clean = normalize_column(df_clean, col)
    
    return df_clean

def save_cleaned_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    raw_data = load_dataset('raw_dataset.csv')
    numeric_cols = ['age', 'income', 'score']
    cleaned_data = clean_dataframe(raw_data, numeric_cols)
    save_cleaned_data(cleaned_data, 'cleaned_dataset.csv')
import pandas as pd
import re

def clean_dataframe(df, columns_to_clean=None, remove_duplicates=True, case_normalization='lower'):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing string columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    columns_to_clean (list, optional): List of column names to clean. If None, all object columns are cleaned.
    remove_duplicates (bool): Whether to remove duplicate rows.
    case_normalization (str): One of 'lower', 'upper', or None for case normalization.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    # Remove duplicates
    if remove_duplicates:
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed = initial_rows - len(df_clean)
        print(f"Removed {removed} duplicate rows.")
    
    # Determine columns to clean
    if columns_to_clean is None:
        columns_to_clean = df_clean.select_dtypes(include=['object']).columns.tolist()
    
    # Clean each column
    for col in columns_to_clean:
        if col in df_clean.columns and df_clean[col].dtype == 'object':
            # Remove extra whitespace
            df_clean[col] = df_clean[col].astype(str).str.strip()
            df_clean[col] = df_clean[col].replace(r'\s+', ' ', regex=True)
            
            # Case normalization
            if case_normalization == 'lower':
                df_clean[col] = df_clean[col].str.lower()
            elif case_normalization == 'upper':
                df_clean[col] = df_clean[col].str.upper()
            
            # Replace empty strings with NaN
            df_clean[col] = df_clean[col].replace(r'^\s*$', pd.NA, regex=True)
    
    return df_clean

def validate_email(email_series):
    """
    Validate email addresses in a pandas Series.
    
    Parameters:
    email_series (pd.Series): Series containing email addresses.
    
    Returns:
    pd.Series: Boolean Series indicating valid emails.
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return email_series.str.match(pattern, na=False)

# Example usage
if __name__ == "__main__":
    # Create sample data
    data = {
        'name': ['John Doe', 'Jane Smith', 'John Doe', '   Alice   '],
        'email': ['john@example.com', 'jane@example', 'john@example.com', 'alice@example.com'],
        'age': [25, 30, 25, 28]
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    # Clean the data
    df_clean = clean_dataframe(df, case_normalization='lower')
    print("Cleaned DataFrame:")
    print(df_clean)
    print("\n")
    
    # Validate emails
    df_clean['valid_email'] = validate_email(df_clean['email'])
    print("DataFrame with email validation:")
    print(df_clean)