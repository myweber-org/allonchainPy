
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, columns=None):
    """
    Remove outliers from DataFrame using Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to process. If None, process all numeric columns.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_clean = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
        df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def normalize_data(df, columns=None, method='minmax'):
    """
    Normalize specified columns in DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to normalize
    method (str): Normalization method - 'minmax' or 'zscore'
    
    Returns:
    pd.DataFrame: DataFrame with normalized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_norm = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if method == 'minmax':
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val != min_val:
                df_norm[col] = (df[col] - min_val) / (max_val - min_val)
        
        elif method == 'zscore':
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val != 0:
                df_norm[col] = (df[col] - mean_val) / std_val
    
    return df_norm

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): Strategy for handling missing values - 'mean', 'median', 'mode', or 'drop'
    columns (list): List of column names to process
    
    Returns:
    pd.DataFrame: DataFrame with handled missing values
    """
    if columns is None:
        columns = df.columns.tolist()
    
    df_handled = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if strategy == 'mean' and pd.api.types.is_numeric_dtype(df[col]):
            df_handled[col].fillna(df[col].mean(), inplace=True)
        
        elif strategy == 'median' and pd.api.types.is_numeric_dtype(df[col]):
            df_handled[col].fillna(df[col].median(), inplace=True)
        
        elif strategy == 'mode':
            df_handled[col].fillna(df[col].mode()[0] if not df[col].mode().empty else None, inplace=True)
        
        elif strategy == 'drop':
            df_handled = df_handled.dropna(subset=[col])
    
    return df_handled.reset_index(drop=True)
import pandas as pd
import numpy as np
from scipy import stats

def load_dataset(filepath):
    """Load dataset from CSV file."""
    return pd.read_csv(filepath)

def remove_outliers_iqr(df, column):
    """Remove outliers using IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def remove_outliers_zscore(df, column, threshold=3):
    """Remove outliers using Z-score method."""
    z_scores = np.abs(stats.zscore(df[column]))
    return df[z_scores < threshold]

def normalize_minmax(df, column):
    """Normalize column using min-max scaling."""
    min_val = df[column].min()
    max_val = df[column].max()
    df[column] = (df[column] - min_val) / (max_val - min_val)
    return df

def normalize_zscore(df, column):
    """Normalize column using Z-score standardization."""
    mean_val = df[column].mean()
    std_val = df[column].std()
    df[column] = (df[column] - mean_val) / std_val
    return df

def handle_missing_values(df, strategy='mean'):
    """Handle missing values with specified strategy."""
    if strategy == 'mean':
        return df.fillna(df.mean())
    elif strategy == 'median':
        return df.fillna(df.median())
    elif strategy == 'mode':
        return df.fillna(df.mode().iloc[0])
    elif strategy == 'drop':
        return df.dropna()
    else:
        raise ValueError("Invalid strategy. Choose from 'mean', 'median', 'mode', or 'drop'.")

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize_method='minmax', missing_strategy='mean'):
    """Complete data cleaning pipeline."""
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in cleaned_df.columns:
            cleaned_df = handle_missing_values(cleaned_df[[column]], strategy=missing_strategy)
            
            if outlier_method == 'iqr':
                cleaned_df = remove_outliers_iqr(cleaned_df, column)
            elif outlier_method == 'zscore':
                cleaned_df = remove_outliers_zscore(cleaned_df, column)
            
            if normalize_method == 'minmax':
                cleaned_df = normalize_minmax(cleaned_df, column)
            elif normalize_method == 'zscore':
                cleaned_df = normalize_zscore(cleaned_df, column)
    
    return cleaned_df

def save_cleaned_data(df, output_path):
    """Save cleaned dataset to CSV."""
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    numeric_cols = ['feature1', 'feature2', 'feature3']
    
    raw_data = load_dataset(input_file)
    cleaned_data = clean_dataset(
        raw_data, 
        numeric_columns=numeric_cols,
        outlier_method='iqr',
        normalize_method='zscore',
        missing_strategy='median'
    )
    
    save_cleaned_data(cleaned_data, output_file)
    print(f"Data cleaning complete. Cleaned data saved to {output_file}")