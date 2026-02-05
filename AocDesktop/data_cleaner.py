
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
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

def calculate_summary_stats(df, column):
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
        'count': len(df[column])
    }
    
    return stats

def process_dataframe(df, numeric_columns):
    """
    Process multiple numeric columns to remove outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of column names to process
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[column]):
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
    
    return cleaned_df
def remove_duplicates_preserve_order(iterable):
    seen = set()
    result = []
    for item in iterable:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Args:
        data (pd.DataFrame): Input DataFrame
        column (str): Column name to process
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    return filtered_data

def calculate_statistics(data, column):
    """
    Calculate basic statistics for a column after outlier removal.
    
    Args:
        data (pd.DataFrame): Input DataFrame
        column (str): Column name to analyze
    
    Returns:
        dict: Dictionary containing statistical measures
    """
    stats = {
        'mean': data[column].mean(),
        'median': data[column].median(),
        'std': data[column].std(),
        'min': data[column].min(),
        'max': data[column].max(),
        'count': data[column].count()
    }
    
    return stats

def process_dataset(data, column):
    """
    Complete pipeline for processing a dataset column.
    
    Args:
        data (pd.DataFrame): Input DataFrame
        column (str): Column name to process
    
    Returns:
        tuple: (cleaned_data, original_stats, cleaned_stats)
    """
    original_stats = calculate_statistics(data, column)
    cleaned_data = remove_outliers_iqr(data, column)
    cleaned_stats = calculate_statistics(cleaned_data, column)
    
    return cleaned_data, original_stats, cleaned_statsimport pandas as pd
import numpy as np

def load_data(filepath):
    """Load dataset from CSV file."""
    return pd.read_csv(filepath)

def remove_outliers(df, column, threshold=3):
    """Remove outliers using Z-score method."""
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    return df[z_scores < threshold]

def normalize_column(df, column):
    """Normalize column values to range [0,1]."""
    min_val = df[column].min()
    max_val = df[column].max()
    df[column] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_data(input_file, output_file):
    """Main data cleaning pipeline."""
    df = load_data(input_file)
    
    if 'value' in df.columns:
        df = remove_outliers(df, 'value')
        df = normalize_column(df, 'value')
    
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")
    return df

if __name__ == "__main__":
    clean_data('raw_data.csv', 'cleaned_data.csv')
import pandas as pd
import numpy as np
from typing import List, Optional

def clean_dataframe(df: pd.DataFrame, 
                    drop_duplicates: bool = True,
                    fill_missing: Optional[str] = None,
                    standardize_columns: bool = True) -> pd.DataFrame:
    """
    Clean a pandas DataFrame by removing duplicates, handling missing values,
    and standardizing column names.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing is not None:
        if fill_missing == 'mean':
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(
                cleaned_df[numeric_cols].mean()
            )
        elif fill_missing == 'median':
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(
                cleaned_df[numeric_cols].median()
            )
        elif fill_missing == 'mode':
            for col in cleaned_df.columns:
                mode_val = cleaned_df[col].mode()
                if not mode_val.empty:
                    cleaned_df[col] = cleaned_df[col].fillna(mode_val[0])
    
    if standardize_columns:
        cleaned_df.columns = (
            cleaned_df.columns.str.strip()
            .str.lower()
            .str.replace(' ', '_')
            .str.replace(r'[^\w_]', '', regex=True)
        )
    
    return cleaned_df

def validate_dataframe(df: pd.DataFrame, 
                       required_columns: List[str] = None) -> bool:
    """
    Validate DataFrame structure and content.
    """
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return False
    
    if df.empty:
        print("DataFrame is empty")
        return False
    
    return True

def remove_outliers_iqr(df: pd.DataFrame, 
                        columns: List[str],
                        multiplier: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers using the Interquartile Range method.
    """
    df_clean = df.copy()
    
    for col in columns:
        if col in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[col]):
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            initial_count = len(df_clean)
            df_clean = df_clean[
                (df_clean[col] >= lower_bound) & 
                (df_clean[col] <= upper_bound)
            ]
            removed = initial_count - len(df_clean)
            print(f"Removed {removed} outliers from column '{col}'")
    
    return df_clean

if __name__ == "__main__":
    sample_data = {
        'Customer ID': [1, 2, 2, 3, 4, 5],
        'Order Value': [100.0, 200.0, 200.0, np.nan, 500.0, 1000.0],
        'Product Category': ['A', 'B', 'B', 'C', 'D', 'E']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned = clean_dataframe(df, fill_missing='mean')
    print(cleaned)