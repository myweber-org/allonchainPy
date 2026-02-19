
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

def clean_dataframe(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df = normalize_column(cleaned_df, col)
    return cleaned_df

def save_cleaned_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    numeric_cols = ['age', 'income', 'score']
    
    try:
        raw_data = load_dataset(input_file)
        cleaned_data = clean_dataframe(raw_data, numeric_cols)
        save_cleaned_data(cleaned_data, output_file)
    except FileNotFoundError:
        print(f"Error: File {input_file} not found")
    except Exception as e:
        print(f"An error occurred: {str(e)}")import pandas as pd
import re

def clean_dataframe(df, text_columns=None):
    """
    Clean a pandas DataFrame by removing duplicates and standardizing text columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        text_columns (list): List of column names to apply text standardization
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Remove duplicate rows
    initial_rows = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    removed_duplicates = initial_rows - len(cleaned_df)
    
    # Standardize text columns if specified
    if text_columns:
        for col in text_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].apply(_standardize_text)
    
    # Remove rows with all NaN values
    cleaned_df = cleaned_df.dropna(how='all')
    
    # Log cleaning results
    print(f"Removed {removed_duplicates} duplicate rows")
    print(f"Final DataFrame shape: {cleaned_df.shape}")
    
    return cleaned_df

def _standardize_text(text):
    """
    Internal function to standardize text by:
    1. Converting to string
    2. Lowercasing
    3. Removing extra whitespace
    4. Removing special characters (keeping alphanumeric and basic punctuation)
    
    Args:
        text: Input text to standardize
    
    Returns:
        str: Standardized text
    """
    if pd.isna(text):
        return text
    
    # Convert to string
    text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove special characters but keep basic punctuation and alphanumeric
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    
    return text

def validate_email_column(df, email_column):
    """
    Validate email addresses in a DataFrame column.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        email_column (str): Name of the column containing email addresses
    
    Returns:
        pd.DataFrame: DataFrame with validation results
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    # Basic email regex pattern
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    # Create validation results
    validation_df = df[[email_column]].copy()
    validation_df['is_valid_email'] = validation_df[email_column].apply(
        lambda x: bool(re.match(email_pattern, str(x))) if pd.notna(x) else False
    )
    validation_df['is_missing'] = validation_df[email_column].isna()
    
    # Calculate statistics
    valid_count = validation_df['is_valid_email'].sum()
    invalid_count = len(validation_df) - valid_count - validation_df['is_missing'].sum()
    missing_count = validation_df['is_missing'].sum()
    
    print(f"Email Validation Results:")
    print(f"  Valid emails: {valid_count}")
    print(f"  Invalid emails: {invalid_count}")
    print(f"  Missing emails: {missing_count}")
    
    return validation_df

# Example usage function
def example_usage():
    """Demonstrate how to use the data cleaning functions."""
    # Create sample data
    data = {
        'name': ['John Doe', 'Jane Smith', 'John Doe', 'Bob Johnson', None],
        'email': ['john@example.com', 'jane@example.com', 'john@example.com', 'invalid-email', None],
        'age': [25, 30, 25, 35, 40]
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the DataFrame
    cleaned_df = clean_dataframe(df, text_columns=['name', 'email'])
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\n" + "="*50 + "\n")
    
    # Validate email column
    validation_results = validate_email_column(cleaned_df, 'email')
    print("Email Validation Results:")
    print(validation_results)

if __name__ == "__main__":
    example_usage()
import pandas as pd
import numpy as np

def clean_dataset(df, drop_threshold=0.5, fill_strategy='mean'):
    """
    Clean a pandas DataFrame by handling missing values and standardizing columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    drop_threshold (float): Threshold for dropping columns with nulls (0 to 1)
    fill_strategy (str): Strategy for filling missing values ('mean', 'median', 'mode')
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    # Create a copy to avoid modifying original
    cleaned_df = df.copy()
    
    # Standardize column names
    cleaned_df.columns = cleaned_df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # Drop columns with too many nulls
    null_ratio = cleaned_df.isnull().sum() / len(cleaned_df)
    cols_to_drop = null_ratio[null_ratio > drop_threshold].index
    cleaned_df = cleaned_df.drop(columns=cols_to_drop)
    
    # Fill remaining missing values
    for col in cleaned_df.columns:
        if cleaned_df[col].isnull().any():
            if cleaned_df[col].dtype in ['int64', 'float64']:
                if fill_strategy == 'mean':
                    fill_value = cleaned_df[col].mean()
                elif fill_strategy == 'median':
                    fill_value = cleaned_df[col].median()
                else:
                    fill_value = cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 0
                cleaned_df[col] = cleaned_df[col].fillna(fill_value)
            else:
                # For categorical columns, fill with most frequent value
                most_frequent = cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 'unknown'
                cleaned_df[col] = cleaned_df[col].fillna(most_frequent)
    
    # Remove duplicate rows
    cleaned_df = cleaned_df.drop_duplicates()
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Check if input is a DataFrame
    if not isinstance(df, pd.DataFrame):
        validation_results['is_valid'] = False
        validation_results['errors'].append('Input is not a pandas DataFrame')
        return validation_results
    
    # Check for empty DataFrame
    if df.empty:
        validation_results['warnings'].append('DataFrame is empty')
    
    # Check required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f'Missing required columns: {missing_columns}')
    
    # Check for all-null columns
    null_columns = df.columns[df.isnull().all()].tolist()
    if null_columns:
        validation_results['warnings'].append(f'Columns with all null values: {null_columns}')
    
    return validation_results

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'Customer ID': [1, 2, 3, None, 5],
        'Order Value': [100.5, None, 75.2, 200.0, 150.0],
        'Product Category': ['A', 'B', None, 'A', 'B'],
        'Region': ['North', 'South', 'East', 'West', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned = clean_dataset(df, drop_threshold=0.3, fill_strategy='mean')
    print("Cleaned DataFrame:")
    print(cleaned)
    print("\n" + "="*50 + "\n")
    
    # Validate the cleaned data
    validation = validate_dataframe(cleaned, required_columns=['customer_id', 'order_value'])
    print("Validation Results:")
    for key, value in validation.items():
        print(f"{key}: {value}")
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
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
    Remove outliers using Z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
    removed_count = len(data) - len(filtered_data)
    
    return filtered_data, removed_count

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].copy()
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].copy()
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(df, numeric_columns=None, outlier_method='iqr', normalize_method='minmax'):
    """
    Comprehensive data cleaning pipeline
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    stats_report = {}
    
    for col in numeric_columns:
        if col not in df.columns:
            continue
            
        original_count = len(cleaned_df)
        
        if outlier_method == 'iqr':
            cleaned_df, removed = remove_outliers_iqr(cleaned_df, col)
        elif outlier_method == 'zscore':
            cleaned_df, removed = remove_outliers_zscore(cleaned_df, col)
        else:
            removed = 0
        
        if normalize_method == 'minmax':
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
        elif normalize_method == 'zscore':
            cleaned_df[col] = normalize_zscore(cleaned_df, col)
        
        stats_report[col] = {
            'original_samples': original_count,
            'removed_outliers': removed,
            'final_samples': len(cleaned_df),
            'normalization_method': normalize_method
        }
    
    return cleaned_df, stats_report

def validate_data(df, required_columns=None, allow_nan=False, min_rows=1):
    """
    Validate dataset structure and content
    """
    if len(df) < min_rows:
        raise ValueError(f"Dataset must have at least {min_rows} rows")
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    if not allow_nan:
        nan_columns = df.columns[df.isnull().any()].tolist()
        if nan_columns:
            raise ValueError(f"NaN values found in columns: {nan_columns}")
    
    return True
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data
import pandas as pd
import hashlib

def remove_duplicates(input_file, output_file, key_columns=None):
    """
    Load a CSV file, remove duplicate rows based on specified columns,
    and save the cleaned data to a new file.
    """
    try:
        df = pd.read_csv(input_file)
        print(f"Loaded data from {input_file}. Shape: {df.shape}")
        
        if key_columns is None:
            key_columns = df.columns.tolist()
        
        initial_count = len(df)
        df_cleaned = df.drop_duplicates(subset=key_columns, keep='first')
        final_count = len(df_cleaned)
        removed_count = initial_count - final_count
        
        df_cleaned.to_csv(output_file, index=False)
        print(f"Removed {removed_count} duplicate rows.")
        print(f"Cleaned data saved to {output_file}. New shape: {df_cleaned.shape}")
        
        return df_cleaned
        
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def generate_data_hash(df):
    """
    Generate a hash for the dataframe to verify data integrity.
    """
    data_string = df.to_string(index=False).encode()
    return hashlib.md5(data_string).hexdigest()

if __name__ == "__main__":
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    
    cleaned_data = remove_duplicates(input_csv, output_csv)
    
    if cleaned_data is not None:
        data_hash = generate_data_hash(cleaned_data)
        print(f"Data integrity hash: {data_hash}")