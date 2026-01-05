import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a DataFrame column using the IQR method.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize a DataFrame column using min-max scaling.
    """
    min_val = data[column].min()
    max_val = data[column].max()
    if max_val - min_val == 0:
        return data[column].apply(lambda x: 0.0)
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize a DataFrame column using z-score normalization.
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    if std_val == 0:
        return data[column].apply(lambda x: 0.0)
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(df, numeric_columns):
    """
    Apply outlier removal and normalization to numeric columns.
    """
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
    return cleaned_df

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': np.random.randn(100) * 10 + 50,
        'B': np.random.randn(100) * 5 + 20,
        'C': np.random.randn(100) * 2 + 10
    })
    print("Original data shape:", sample_data.shape)
    cleaned = clean_dataset(sample_data, ['A', 'B', 'C'])
    print("Cleaned data shape:", cleaned.shape)
    print("Cleaned data summary:")
    print(cleaned.describe())
import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None):
    """Remove duplicate rows from a DataFrame."""
    if subset is None:
        subset = df.columns.tolist()
    return df.drop_duplicates(subset=subset, keep='first')

def fill_missing_values(df, strategy='mean', columns=None):
    """Fill missing values in specified columns using a given strategy."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_filled = df.copy()
    
    for col in columns:
        if col in df.columns:
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mode':
                fill_value = df[col].mode()[0] if not df[col].mode().empty else np.nan
            elif strategy == 'constant':
                fill_value = 0
            else:
                raise ValueError(f"Unsupported strategy: {strategy}")
            
            df_filled[col].fillna(fill_value, inplace=True)
    
    return df_filled

def validate_numeric_range(df, column, min_val=None, max_val=None):
    """Validate that values in a column are within a specified range."""
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame")
    
    mask = pd.Series(True, index=df.index)
    
    if min_val is not None:
        mask = mask & (df[column] >= min_val)
    
    if max_val is not None:
        mask = mask & (df[column] <= max_val)
    
    invalid_count = (~mask).sum()
    valid_percentage = (mask.sum() / len(df)) * 100
    
    return {
        'is_valid': invalid_count == 0,
        'invalid_count': invalid_count,
        'valid_percentage': valid_percentage,
        'invalid_indices': df.index[~mask].tolist()
    }

def normalize_column(df, column, method='minmax'):
    """Normalize values in a column using specified method."""
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame")
    
    if method == 'minmax':
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val == min_val:
            return df[column]
        return (df[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = df[column].mean()
        std_val = df[column].std()
        if std_val == 0:
            return df[column]
        return (df[column] - mean_val) / std_val
    
    else:
        raise ValueError(f"Unsupported normalization method: {method}")

def clean_dataframe(df, 
                    remove_dups=True, 
                    fill_na=True, 
                    fill_strategy='mean',
                    validation_rules=None):
    """Apply multiple cleaning operations to a DataFrame."""
    cleaned_df = df.copy()
    
    if remove_dups:
        cleaned_df = remove_duplicates(cleaned_df)
    
    if fill_na:
        cleaned_df = fill_missing_values(cleaned_df, strategy=fill_strategy)
    
    if validation_rules:
        for rule in validation_rules:
            column = rule.get('column')
            min_val = rule.get('min')
            max_val = rule.get('max')
            
            if column:
                result = validate_numeric_range(cleaned_df, column, min_val, max_val)
                if not result['is_valid']:
                    print(f"Warning: {result['invalid_count']} invalid values found in column '{column}'")
    
    return cleaned_df
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_iqr(self, columns=None, threshold=1.5):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_clean = self.df.copy()
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                mask = (self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)
                df_clean = df_clean[mask]
        
        removed_count = self.original_shape[0] - df_clean.shape[0]
        self.df = df_clean.reset_index(drop=True)
        return removed_count
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_normalized = self.df.copy()
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:
                    df_normalized[col] = (self.df[col] - min_val) / (max_val - min_val)
        
        self.df = df_normalized
        return self
    
    def fill_missing_median(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
            
        df_filled = self.df.copy()
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                median_val = self.df[col].median()
                df_filled[col] = self.df[col].fillna(median_val)
        
        self.df = df_filled
        return self
    
    def get_clean_data(self):
        return self.df.copy()
    
    def get_summary(self):
        summary = {
            'original_rows': self.original_shape[0],
            'current_rows': self.df.shape[0],
            'columns': self.df.shape[1],
            'missing_values': self.df.isnull().sum().sum(),
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(exclude=[np.number]).columns)
        }
        return summary

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.randint(1, 100, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    df = pd.DataFrame(data)
    df.loc[np.random.choice(df.index, 50), 'feature_a'] = np.nan
    df.loc[np.random.choice(df.index, 20), 'feature_b'] = 9999
    
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    cleaner = DataCleaner(sample_df)
    
    print("Initial summary:")
    print(cleaner.get_summary())
    
    outliers_removed = cleaner.remove_outliers_iqr(['feature_a', 'feature_b'])
    print(f"\nRemoved {outliers_removed} outliers")
    
    cleaner.fill_missing_median()
    cleaner.normalize_minmax(['feature_a', 'feature_b', 'feature_c'])
    
    print("\nFinal summary:")
    print(cleaner.get_summary())
    
    clean_data = cleaner.get_clean_data()
    print(f"\nClean data shape: {clean_data.shape}")
    print(f"Clean data preview:\n{clean_data.head()}")
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, threshold=1.5):
    """
    Remove outliers from a DataFrame column using IQR method.
    
    Args:
        dataframe: pandas DataFrame
        column: column name to process
        threshold: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    
    return filtered_df

def normalize_column(dataframe, column, method='minmax'):
    """
    Normalize a column using specified method.
    
    Args:
        dataframe: pandas DataFrame
        column: column name to normalize
        method: 'minmax' or 'zscore'
    
    Returns:
        Series with normalized values
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if method == 'minmax':
        min_val = dataframe[column].min()
        max_val = dataframe[column].max()
        if max_val == min_val:
            return pd.Series([0] * len(dataframe), index=dataframe.index)
        normalized = (dataframe[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = dataframe[column].mean()
        std_val = dataframe[column].std()
        if std_val == 0:
            return pd.Series([0] * len(dataframe), index=dataframe.index)
        normalized = (dataframe[column] - mean_val) / std_val
    
    else:
        raise ValueError("Method must be 'minmax' or 'zscore'")
    
    return normalized

def clean_dataset(dataframe, numeric_columns=None, outlier_threshold=1.5, normalize_method='minmax'):
    """
    Clean dataset by removing outliers and normalizing numeric columns.
    
    Args:
        dataframe: pandas DataFrame
        numeric_columns: list of numeric columns to process (default: all numeric columns)
        outlier_threshold: IQR threshold for outlier removal
        normalize_method: normalization method
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = dataframe.copy()
    
    for column in numeric_columns:
        if column in cleaned_df.columns:
            # Remove outliers
            q1 = cleaned_df[column].quantile(0.25)
            q3 = cleaned_df[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - outlier_threshold * iqr
            upper_bound = q3 + outlier_threshold * iqr
            
            mask = (cleaned_df[column] >= lower_bound) & (cleaned_df[column] <= upper_bound)
            cleaned_df = cleaned_df[mask]
            
            # Normalize
            cleaned_df[column] = normalize_column(cleaned_df, column, normalize_method)
    
    return cleaned_df.reset_index(drop=True)

def detect_skewed_columns(dataframe, threshold=0.5):
    """
    Detect skewed columns based on absolute skewness value.
    
    Args:
        dataframe: pandas DataFrame
        threshold: skewness threshold (default 0.5)
    
    Returns:
        Dictionary with column names and their skewness values
    """
    skewed_columns = {}
    
    for column in dataframe.select_dtypes(include=[np.number]).columns:
        skewness = stats.skew(dataframe[column].dropna())
        if abs(skewness) > threshold:
            skewed_columns[column] = skewness
    
    return skewed_columns

def log_transform_skewed(dataframe, skewed_columns):
    """
    Apply log transformation to skewed columns.
    
    Args:
        dataframe: pandas DataFrame
        skewed_columns: list of columns to transform
    
    Returns:
        DataFrame with transformed columns
    """
    transformed_df = dataframe.copy()
    
    for column in skewed_columns:
        if column in transformed_df.columns:
            # Add small constant to handle zero or negative values
            min_val = transformed_df[column].min()
            if min_val <= 0:
                constant = abs(min_val) + 1
                transformed_df[column] = np.log(transformed_df[column] + constant)
            else:
                transformed_df[column] = np.log(transformed_df[column])
    
    return transformed_df
import pandas as pd
import re

def clean_dataframe(df, columns_to_normalize=None, remove_duplicates=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing string columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        columns_to_normalize (list, optional): List of column names to normalize.
            If None, all object dtype columns will be normalized.
        remove_duplicates (bool): Whether to remove duplicate rows.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    if remove_duplicates:
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed = initial_rows - len(df_clean)
        print(f"Removed {removed} duplicate rows.")
    
    if columns_to_normalize is None:
        columns_to_normalize = df_clean.select_dtypes(include=['object']).columns.tolist()
    
    for col in columns_to_normalize:
        if col in df_clean.columns and df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].apply(_normalize_string)
            print(f"Normalized column: {col}")
    
    return df_clean

def _normalize_string(text):
    """
    Normalize a string by converting to lowercase, removing extra whitespace,
    and stripping special characters.
    
    Args:
        text (str): Input string.
    
    Returns:
        str: Normalized string.
    """
    if pd.isna(text):
        return text
    
    text = str(text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = re.sub(r'[^\w\s-]', '', text)
    
    return text

def validate_email_column(df, email_column):
    """
    Validate email addresses in a DataFrame column.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        email_column (str): Name of the column containing email addresses.
    
    Returns:
        pd.DataFrame: DataFrame with an additional 'email_valid' boolean column.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame.")
    
    df_valid = df.copy()
    email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    df_valid['email_valid'] = df_valid[email_column].apply(
        lambda x: bool(re.match(email_regex, str(x))) if pd.notna(x) else False
    )
    
    valid_count = df_valid['email_valid'].sum()
    print(f"Found {valid_count} valid email addresses out of {len(df_valid)} rows.")
    
    return df_valid