
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

def standardize_zscore(data, column):
    mean_val = data[column].mean()
    std_val = data[column].std()
    data[column + '_standardized'] = (data[column] - mean_val) / std_val
    return data

def clean_dataset(df, numeric_columns, method='iqr', normalize=False):
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        elif method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, col)
        
        if normalize:
            cleaned_df = normalize_minmax(cleaned_df, col)
        else:
            cleaned_df = standardize_zscore(cleaned_df, col)
    
    return cleaned_df

def validate_data(df, required_columns):
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    numeric_check = df[required_columns].select_dtypes(include=[np.number])
    if len(numeric_check.columns) != len(required_columns):
        raise ValueError("All specified columns must be numeric")
    
    return True
import pandas as pd
import re

def clean_dataframe(df, columns_to_clean=None, remove_duplicates=True, case_normalization='lower'):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing string columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns_to_clean (list): List of column names to clean. If None, clean all object columns.
    remove_duplicates (bool): Whether to remove duplicate rows
    case_normalization (str): 'lower', 'upper', or None for case normalization
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if remove_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if columns_to_clean is None:
        columns_to_clean = cleaned_df.select_dtypes(include=['object']).columns.tolist()
    
    for column in columns_to_clean:
        if column in cleaned_df.columns and cleaned_df[column].dtype == 'object':
            cleaned_df[column] = cleaned_df[column].astype(str)
            
            if case_normalization == 'lower':
                cleaned_df[column] = cleaned_df[column].str.lower()
            elif case_normalization == 'upper':
                cleaned_df[column] = cleaned_df[column].str.upper()
            
            cleaned_df[column] = cleaned_df[column].str.strip()
            cleaned_df[column] = cleaned_df[column].replace(r'\s+', ' ', regex=True)
    
    return cleaned_df

def validate_email_column(df, email_column):
    """
    Validate email addresses in a DataFrame column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    email_column (str): Name of the column containing email addresses
    
    Returns:
    pd.DataFrame: DataFrame with additional 'email_valid' column
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    validated_df = df.copy()
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    validated_df['email_valid'] = validated_df[email_column].apply(
        lambda x: bool(re.match(email_pattern, str(x))) if pd.notnull(x) else False
    )
    
    valid_count = validated_df['email_valid'].sum()
    total_count = len(validated_df)
    print(f"Valid emails: {valid_count}/{total_count} ({valid_count/total_count*100:.1f}%)")
    
    return validated_df

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers from a numeric column using the IQR method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Name of the numeric column
    multiplier (float): IQR multiplier for outlier detection
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' is not numeric")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    initial_rows = len(df)
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    removed = initial_rows - len(filtered_df)
    
    print(f"Removed {removed} outliers from column '{column}'")
    print(f"Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    return filtered_df

def save_cleaned_data(df, output_path, format='csv'):
    """
    Save cleaned DataFrame to file.
    
    Parameters:
    df (pd.DataFrame): DataFrame to save
    output_path (str): Output file path
    format (str): 'csv' or 'parquet'
    """
    if format == 'csv':
        df.to_csv(output_path, index=False)
    elif format == 'parquet':
        df.to_parquet(output_path, index=False)
    else:
        raise ValueError("Format must be 'csv' or 'parquet'")
    
    print(f"Data saved to {output_path} ({format.upper()})")
import pandas as pd
import numpy as np

def clean_dataset(df, columns_to_check=None, fill_missing=True):
    """
    Clean dataset by removing duplicates and handling missing values.
    
    Args:
        df: pandas DataFrame to clean
        columns_to_check: list of columns to check for duplicates (default: all columns)
        fill_missing: boolean indicating whether to fill missing values (default: True)
    
    Returns:
        Cleaned pandas DataFrame
    """
    original_rows = df.shape[0]
    
    # Remove duplicates
    if columns_to_check is None:
        df_cleaned = df.drop_duplicates()
    else:
        df_cleaned = df.drop_duplicates(subset=columns_to_check)
    
    removed_duplicates = original_rows - df_cleaned.shape[0]
    
    # Handle missing values
    if fill_missing:
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        categorical_cols = df_cleaned.select_dtypes(exclude=[np.number]).columns
        
        # Fill numeric columns with median
        for col in numeric_cols:
            if df_cleaned[col].isnull().any():
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
        
        # Fill categorical columns with mode
        for col in categorical_cols:
            if df_cleaned[col].isnull().any():
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])
    
    missing_before = df.isnull().sum().sum()
    missing_after = df_cleaned.isnull().sum().sum()
    
    print(f"Original dataset: {original_rows} rows")
    print(f"Removed duplicates: {removed_duplicates} rows")
    print(f"Cleaned dataset: {df_cleaned.shape[0]} rows")
    print(f"Missing values before: {missing_before}")
    print(f"Missing values after: {missing_after}")
    
    return df_cleaned

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate dataset structure and content.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: list of required column names
        min_rows: minimum number of rows required
    
    Returns:
        boolean indicating if validation passed
    """
    if df.empty:
        print("Validation failed: DataFrame is empty")
        return False
    
    if df.shape[0] < min_rows:
        print(f"Validation failed: Less than {min_rows} rows")
        return False
    
    if required_columns is not None:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Validation failed: Missing columns: {missing_columns}")
            return False
    
    print("Validation passed")
    return True

def save_cleaned_data(df, output_path, format='csv'):
    """
    Save cleaned dataset to file.
    
    Args:
        df: pandas DataFrame to save
        output_path: path to save the file
        format: file format ('csv' or 'parquet')
    """
    if format == 'csv':
        df.to_csv(output_path, index=False)
    elif format == 'parquet':
        df.to_parquet(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', None, 'Eve'],
        'age': [25, 30, 30, None, 35, 40],
        'score': [85.5, 92.0, 92.0, 78.5, 88.0, 95.5]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned_df = clean_dataset(df, columns_to_check=['id', 'name'])
    
    print("\n" + "="*50 + "\n")
    print("Cleaned DataFrame:")
    print(cleaned_df)
    
    # Validate the cleaned data
    validation_passed = validate_data(cleaned_df, required_columns=['id', 'name', 'age', 'score'])
    
    if validation_passed:
        # Save the cleaned data
        save_cleaned_data(cleaned_df, 'cleaned_data.csv')