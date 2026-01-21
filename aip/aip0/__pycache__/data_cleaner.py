
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas DataFrame column using the IQR method.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame.
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
    data (pd.DataFrame): The input DataFrame.
    column (str): The column name to analyze.
    
    Returns:
    dict: Dictionary containing mean, median, and standard deviation.
    """
    stats = {
        'mean': data[column].mean(),
        'median': data[column].median(),
        'std': data[column].std()
    }
    return stats
import pandas as pd
import numpy as np
from scipy import stats

def load_dataset(file_path):
    """Load dataset from CSV file."""
    return pd.read_csv(file_path)

def remove_outliers_iqr(df, columns):
    """Remove outliers using IQR method."""
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
    """Normalize specified columns."""
    df_norm = df.copy()
    for col in columns:
        if method == 'minmax':
            df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())
        elif method == 'zscore':
            df_norm[col] = stats.zscore(df_norm[col])
        else:
            raise ValueError("Method must be 'minmax' or 'zscore'")
    return df_norm

def handle_missing_values(df, strategy='mean'):
    """Handle missing values with specified strategy."""
    df_filled = df.copy()
    numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if strategy == 'mean':
            df_filled[col].fillna(df_filled[col].mean(), inplace=True)
        elif strategy == 'median':
            df_filled[col].fillna(df_filled[col].median(), inplace=True)
        elif strategy == 'mode':
            df_filled[col].fillna(df_filled[col].mode()[0], inplace=True)
        else:
            raise ValueError("Strategy must be 'mean', 'median', or 'mode'")
    
    return df_filled

def clean_dataset(input_path, output_path, numeric_columns):
    """Complete data cleaning pipeline."""
    df = load_dataset(input_path)
    df = handle_missing_values(df, strategy='median')
    df = remove_outliers_iqr(df, numeric_columns)
    df = normalize_data(df, numeric_columns, method='minmax')
    df.to_csv(output_path, index=False)
    return df

if __name__ == "__main__":
    numeric_cols = ['age', 'income', 'score']
    cleaned_df = clean_dataset('raw_data.csv', 'cleaned_data.csv', numeric_cols)
    print(f"Data cleaning complete. Shape: {cleaned_df.shape}")import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, columns=None):
    """
    Normalize data using min-max scaling
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    normalized_data = data.copy()
    
    for col in columns:
        if col in data.columns and np.issubdtype(data[col].dtype, np.number):
            min_val = data[col].min()
            max_val = data[col].max()
            
            if max_val > min_val:
                normalized_data[col] = (data[col] - min_val) / (max_val - min_val)
    
    return normalized_data

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values with specified strategy
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    cleaned_data = data.copy()
    
    for col in columns:
        if col in data.columns and data[col].isnull().any():
            if strategy == 'mean':
                fill_value = data[col].mean()
            elif strategy == 'median':
                fill_value = data[col].median()
            elif strategy == 'mode':
                fill_value = data[col].mode()[0]
            elif strategy == 'zero':
                fill_value = 0
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            cleaned_data[col].fillna(fill_value, inplace=True)
    
    return cleaned_data

def clean_dataset(data, outlier_columns=None, normalize_columns=None, missing_strategy='mean'):
    """
    Comprehensive data cleaning pipeline
    """
    cleaned_data = data.copy()
    
    if outlier_columns:
        for col in outlier_columns:
            if col in cleaned_data.columns:
                cleaned_data = remove_outliers_iqr(cleaned_data, col)
    
    cleaned_data = handle_missing_values(cleaned_data, strategy=missing_strategy, columns=normalize_columns)
    
    if normalize_columns:
        cleaned_data = normalize_minmax(cleaned_data, columns=normalize_columns)
    
    return cleaned_data
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
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
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

def clean_dataset(df, columns_to_clean=None):
    """
    Clean multiple columns in a DataFrame by removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns_to_clean (list): List of column names to clean. If None, clean all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    dict: Dictionary of summary statistics for each cleaned column
    """
    if columns_to_clean is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns_to_clean = list(numeric_cols)
    
    cleaned_df = df.copy()
    summary_stats = {}
    
    for column in columns_to_clean:
        if column in cleaned_df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            
            stats = calculate_summary_statistics(cleaned_df, column)
            stats['outliers_removed'] = removed_count
            summary_stats[column] = stats
    
    return cleaned_df, summary_stats

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[::100, 'A'] = 500
    
    print("Original DataFrame shape:", df.shape)
    print("\nOriginal summary statistics:")
    print(df.describe())
    
    cleaned_df, stats = clean_dataset(df, ['A', 'B'])
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("\nCleaned summary statistics:")
    print(cleaned_df.describe())
    
    print("\nDetailed statistics per column:")
    for col, col_stats in stats.items():
        print(f"\n{col}:")
        for stat_name, stat_value in col_stats.items():
            print(f"  {stat_name}: {stat_value:.2f}")
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to remove duplicate rows
        fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
        print("Dropped rows with missing values")
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        print(f"Filled missing numeric values with {fill_missing}")
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 'Unknown')
        print("Filled missing categorical values with mode")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'issues': [],
        'summary': {
            'rows': len(df),
            'columns': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'duplicates': df.duplicated().sum()
        }
    }
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Missing required columns: {missing_cols}")
    
    if df.empty:
        validation_results['is_valid'] = False
        validation_results['issues'].append("DataFrame is empty")
    
    return validation_results

def normalize_columns(df, columns=None):
    """
    Normalize specified columns to range [0, 1].
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): Columns to normalize, defaults to all numeric columns
    
    Returns:
        pd.DataFrame: DataFrame with normalized columns
    """
    normalized_df = df.copy()
    
    if columns is None:
        columns = normalized_df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in normalized_df.columns and normalized_df[col].dtype in [np.float64, np.int64]:
            col_min = normalized_df[col].min()
            col_max = normalized_df[col].max()
            if col_max != col_min:
                normalized_df[col] = (normalized_df[col] - col_min) / (col_max - col_min)
    
    return normalized_df

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, 5, None],
        'B': [10, 20, 20, 40, 50, 60],
        'C': ['X', 'Y', 'Y', 'Z', None, 'X']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned = clean_dataset(df, fill_missing='mean')
    print("Cleaned DataFrame:")
    print(cleaned)
    print("\n" + "="*50 + "\n")
    
    validation = validate_dataframe(cleaned, required_columns=['A', 'B', 'C'])
    print("Validation Results:")
    print(validation)
    print("\n" + "="*50 + "\n")
    
    normalized = normalize_columns(cleaned, columns=['A', 'B'])
    print("Normalized DataFrame:")
    print(normalized)