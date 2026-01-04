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