
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

def handle_missing_values(data, strategy='mean'):
    if strategy == 'mean':
        return data.fillna(data.mean())
    elif strategy == 'median':
        return data.fillna(data.median())
    elif strategy == 'mode':
        return data.fillna(data.mode().iloc[0])
    elif strategy == 'drop':
        return data.dropna()
    else:
        raise ValueError("Invalid strategy. Choose from 'mean', 'median', 'mode', or 'drop'")

def clean_dataset(data, numeric_columns, outlier_method='iqr', normalize_method='minmax', missing_strategy='mean'):
    cleaned_data = data.copy()
    
    for col in numeric_columns:
        if outlier_method == 'iqr':
            cleaned_data = remove_outliers_iqr(cleaned_data, col)
        elif outlier_method == 'zscore':
            cleaned_data = remove_outliers_zscore(cleaned_data, col)
    
    cleaned_data = handle_missing_values(cleaned_data, strategy=missing_strategy)
    
    for col in numeric_columns:
        if normalize_method == 'minmax':
            cleaned_data = normalize_minmax(cleaned_data, col)
        elif normalize_method == 'zscore':
            cleaned_data = normalize_zscore(cleaned_data, col)
    
    return cleaned_data

def validate_data(data, numeric_columns):
    validation_report = {}
    
    for col in numeric_columns:
        validation_report[col] = {
            'missing_count': data[col].isnull().sum(),
            'missing_percentage': (data[col].isnull().sum() / len(data)) * 100,
            'mean': data[col].mean(),
            'std': data[col].std(),
            'min': data[col].min(),
            'max': data[col].max(),
            'skewness': data[col].skew(),
            'kurtosis': data[col].kurtosis()
        }
    
    return pd.DataFrame(validation_report).T

def export_cleaned_data(data, filename, format='csv'):
    if format == 'csv':
        data.to_csv(filename, index=False)
    elif format == 'excel':
        data.to_excel(filename, index=False)
    elif format == 'json':
        data.to_json(filename, orient='records')
    else:
        raise ValueError("Invalid format. Choose from 'csv', 'excel', or 'json'")
import pandas as pd
import numpy as np
from typing import List, Optional

def clean_dataframe(df: pd.DataFrame, 
                    drop_duplicates: bool = True,
                    columns_to_standardize: Optional[List[str]] = None,
                    date_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Clean a pandas DataFrame by removing duplicates, standardizing text columns,
    and converting date columns to datetime format.
    """
    df_clean = df.copy()
    
    if drop_duplicates:
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed = initial_rows - len(df_clean)
        print(f"Removed {removed} duplicate rows")
    
    if columns_to_standardize:
        for col in columns_to_standardize:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()
                df_clean[col] = df_clean[col].replace(['nan', 'none', ''], np.nan)
    
    if date_columns:
        for col in date_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
    
    missing_values = df_clean.isnull().sum().sum()
    if missing_values > 0:
        print(f"Warning: DataFrame contains {missing_values} missing values")
    
    return df_clean

def validate_email_column(df: pd.DataFrame, email_column: str) -> pd.Series:
    """
    Validate email addresses in a specified column and return a boolean series.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return df[email_column].str.match(email_pattern, na=False)

def remove_outliers_iqr(df: pd.DataFrame, 
                       numeric_columns: List[str],
                       multiplier: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers from numeric columns using the Interquartile Range method.
    """
    df_filtered = df.copy()
    
    for col in numeric_columns:
        if col in df_filtered.columns and pd.api.types.is_numeric_dtype(df_filtered[col]):
            Q1 = df_filtered[col].quantile(0.25)
            Q3 = df_filtered[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            initial_count = len(df_filtered)
            df_filtered = df_filtered[(df_filtered[col] >= lower_bound) & 
                                     (df_filtered[col] <= upper_bound)]
            removed = initial_count - len(df_filtered)
            print(f"Removed {removed} outliers from column '{col}'")
    
    return df_filtered

if __name__ == "__main__":
    sample_data = {
        'name': ['Alice', 'Bob', 'Alice', 'Charlie', 'David', 'Eve'],
        'email': ['alice@example.com', 'bob@test.org', 'alice@example.com', 
                  'invalid-email', 'david@company.co', 'eve@sample.net'],
        'age': [25, 30, 25, 35, 150, 28],
        'join_date': ['2023-01-15', '2023-02-20', '2023-01-15', 
                      'invalid-date', '2023-03-10', '2023-04-05']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned_df = clean_dataframe(
        df,
        columns_to_standardize=['name'],
        date_columns=['join_date']
    )
    
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\n" + "="*50 + "\n")
    
    valid_emails = validate_email_column(cleaned_df, 'email')
    print("Valid emails:")
    print(valid_emails)
    print("\n" + "="*50 + "\n")
    
    filtered_df = remove_outliers_iqr(cleaned_df, ['age'])
    print("DataFrame after outlier removal:")
    print(filtered_df)