import pandas as pd

def clean_dataset(df, remove_duplicates=True, fill_method=None):
    """
    Clean a pandas DataFrame by handling missing values and removing duplicates.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        remove_duplicates (bool): Whether to remove duplicate rows
        fill_method (str or None): Method to fill missing values - 'mean', 'median', 'mode', or None
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if fill_method:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        
        if fill_method == 'mean':
            for col in numeric_cols:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
        elif fill_method == 'median':
            for col in numeric_cols:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        elif fill_method == 'mode':
            for col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
    else:
        # Drop rows with all NaN values
        cleaned_df = cleaned_df.dropna(how='all')
    
    # Remove duplicates
    if remove_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate a DataFrame for basic integrity checks.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        dict: Dictionary with validation results
    """
    validation_results = {
        'is_valid': True,
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'column_types': df.dtypes.to_dict()
    }
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        validation_results['missing_required_columns'] = missing_columns
        validation_results['is_valid'] = len(missing_columns) == 0
    
    return validation_results

def sample_data(df, sample_size=0.1, random_state=42):
    """
    Create a random sample from the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        sample_size (float or int): Fraction or number of samples
        random_state (int): Random seed for reproducibility
    
    Returns:
        pd.DataFrame: Sampled DataFrame
    """
    if isinstance(sample_size, float) and 0 < sample_size < 1:
        return df.sample(frac=sample_size, random_state=random_state)
    elif isinstance(sample_size, int) and sample_size > 0:
        return df.sample(n=min(sample_size, len(df)), random_state=random_state)
    else:
        raise ValueError("sample_size must be a float between 0 and 1 or a positive integer")import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
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
    Calculate summary statistics for a column.
    
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
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to clean. If None, clean all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            original_len = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            removed_count = original_len - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{col}'")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[np.random.choice(df.index, 50), 'A'] = np.random.uniform(500, 1000, 50)
    
    print("Original data shape:", df.shape)
    print("\nOriginal summary for column 'A':")
    print(calculate_summary_stats(df, 'A'))
    
    cleaned_df = clean_numeric_data(df, ['A', 'B'])
    
    print("\nCleaned data shape:", cleaned_df.shape)
    print("\nCleaned summary for column 'A':")
    print(calculate_summary_stats(cleaned_df, 'A'))
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, threshold=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
    return filtered_data

def normalize_minmax(data, columns=None):
    """
    Normalize data using Min-Max scaling
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    normalized_data = data.copy()
    for col in columns:
        if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
            min_val = data[col].min()
            max_val = data[col].max()
            if max_val != min_val:
                normalized_data[col] = (data[col] - min_val) / (max_val - min_val)
    
    return normalized_data

def normalize_zscore(data, columns=None):
    """
    Normalize data using Z-score standardization
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

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in DataFrame
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns
    
    processed_data = data.copy()
    
    for col in columns:
        if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
            if strategy == 'mean':
                fill_value = data[col].mean()
            elif strategy == 'median':
                fill_value = data[col].median()
            elif strategy == 'mode':
                fill_value = data[col].mode()[0] if not data[col].mode().empty else 0
            elif strategy == 'zero':
                fill_value = 0
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            processed_data[col] = data[col].fillna(fill_value)
    
    return processed_data

def clean_dataset(data, outlier_method='iqr', normalize_method='minmax', 
                  missing_strategy='mean', outlier_threshold=1.5):
    """
    Complete data cleaning pipeline
    """
    cleaned_data = data.copy()
    
    numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if outlier_method == 'iqr':
            cleaned_data = remove_outliers_iqr(cleaned_data, col, outlier_threshold)
        elif outlier_method == 'zscore':
            cleaned_data = remove_outliers_zscore(cleaned_data, col, outlier_threshold)
    
    cleaned_data = handle_missing_values(cleaned_data, strategy=missing_strategy)
    
    if normalize_method == 'minmax':
        cleaned_data = normalize_minmax(cleaned_data)
    elif normalize_method == 'zscore':
        cleaned_data = normalize_zscore(cleaned_data)
    
    return cleaned_data
import csv
import re
from typing import List, Dict, Optional

def clean_string(value: str) -> str:
    """Remove extra whitespace and normalize string."""
    if not isinstance(value, str):
        return str(value)
    cleaned = re.sub(r'\s+', ' ', value.strip())
    return cleaned

def validate_email(email: str) -> bool:
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def read_csv_file(filepath: str) -> List[Dict]:
    """Read CSV file and return list of dictionaries."""
    data = []
    try:
        with open(filepath, 'r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
    except Exception as e:
        print(f"Error reading CSV: {e}")
    return data

def clean_csv_data(data: List[Dict], columns_to_clean: Optional[List[str]] = None) -> List[Dict]:
    """Clean specified columns in CSV data."""
    cleaned_data = []
    for row in data:
        cleaned_row = {}
        for key, value in row.items():
            if columns_to_clean is None or key in columns_to_clean:
                if isinstance(value, str):
                    cleaned_row[key] = clean_string(value)
                else:
                    cleaned_row[key] = value
            else:
                cleaned_row[key] = value
        cleaned_data.append(cleaned_row)
    return cleaned_data

def write_csv_file(data: List[Dict], filepath: str) -> bool:
    """Write data to CSV file."""
    if not data:
        return False
    
    try:
        fieldnames = data[0].keys()
        with open(filepath, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        return True
    except Exception as e:
        print(f"Error writing CSV: {e}")
        return False

def filter_invalid_emails(data: List[Dict], email_column: str) -> List[Dict]:
    """Filter out rows with invalid email addresses."""
    valid_data = []
    invalid_data = []
    
    for row in data:
        if email_column in row and validate_email(row[email_column]):
            valid_data.append(row)
        else:
            invalid_data.append(row)
    
    if invalid_data:
        print(f"Filtered out {len(invalid_data)} rows with invalid emails.")
    
    return valid_data

def process_csv(input_file: str, output_file: str, email_column: Optional[str] = None) -> None:
    """Main function to process CSV file."""
    print(f"Processing {input_file}...")
    
    data = read_csv_file(input_file)
    if not data:
        print("No data to process.")
        return
    
    cleaned_data = clean_csv_data(data)
    
    if email_column:
        cleaned_data = filter_invalid_emails(cleaned_data, email_column)
    
    if write_csv_file(cleaned_data, output_file):
        print(f"Successfully processed {len(cleaned_data)} rows.")
        print(f"Output saved to {output_file}")
    else:
        print("Failed to write output file.")

if __name__ == "__main__":
    process_csv("input.csv", "output.csv", email_column="email")