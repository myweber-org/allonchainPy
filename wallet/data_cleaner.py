
import pandas as pd
import numpy as np
from scipy import stats

def load_dataset(filepath):
    """Load dataset from CSV file"""
    return pd.read_csv(filepath)

def remove_outliers_iqr(df, columns):
    """Remove outliers using IQR method"""
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def remove_outliers_zscore(df, columns, threshold=3):
    """Remove outliers using Z-score method"""
    df_clean = df.copy()
    for col in columns:
        z_scores = np.abs(stats.zscore(df_clean[col]))
        df_clean = df_clean[z_scores < threshold]
    return df_clean

def normalize_minmax(df, columns):
    """Normalize data using Min-Max scaling"""
    df_normalized = df.copy()
    for col in columns:
        min_val = df_normalized[col].min()
        max_val = df_normalized[col].max()
        df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
    return df_normalized

def normalize_zscore(df, columns):
    """Normalize data using Z-score standardization"""
    df_normalized = df.copy()
    for col in columns:
        mean_val = df_normalized[col].mean()
        std_val = df_normalized[col].std()
        df_normalized[col] = (df_normalized[col] - mean_val) / std_val
    return df_normalized

def handle_missing_values(df, strategy='mean'):
    """Handle missing values with specified strategy"""
    df_filled = df.copy()
    numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
    
    if strategy == 'mean':
        for col in numeric_cols:
            df_filled[col].fillna(df_filled[col].mean(), inplace=True)
    elif strategy == 'median':
        for col in numeric_cols:
            df_filled[col].fillna(df_filled[col].median(), inplace=True)
    elif strategy == 'mode':
        for col in numeric_cols:
            df_filled[col].fillna(df_filled[col].mode()[0], inplace=True)
    elif strategy == 'drop':
        df_filled.dropna(inplace=True)
    
    return df_filled

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize_method='minmax', missing_strategy='mean'):
    """Complete data cleaning pipeline"""
    print(f"Original dataset shape: {df.shape}")
    
    df_clean = handle_missing_values(df, strategy=missing_strategy)
    print(f"After handling missing values: {df_clean.shape}")
    
    if outlier_method == 'iqr':
        df_clean = remove_outliers_iqr(df_clean, numeric_columns)
    elif outlier_method == 'zscore':
        df_clean = remove_outliers_zscore(df_clean, numeric_columns)
    print(f"After outlier removal: {df_clean.shape}")
    
    if normalize_method == 'minmax':
        df_clean = normalize_minmax(df_clean, numeric_columns)
    elif normalize_method == 'zscore':
        df_clean = normalize_zscore(df_clean, numeric_columns)
    
    print(f"Final cleaned dataset shape: {df_clean.shape}")
    return df_clean

def save_cleaned_data(df, output_path):
    """Save cleaned dataset to CSV"""
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    try:
        data = load_dataset(input_file)
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        cleaned_data = clean_dataset(
            data, 
            numeric_columns=numeric_cols,
            outlier_method='iqr',
            normalize_method='zscore',
            missing_strategy='mean'
        )
        
        save_cleaned_data(cleaned_data, output_file)
        
        print("\nCleaning summary:")
        print(f"Original records: {len(data)}")
        print(f"Cleaned records: {len(cleaned_data)}")
        print(f"Records removed: {len(data) - len(cleaned_data)}")
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
import pandas as pd
import re

def clean_dataframe(df, column_names=None):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing string columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        column_names (list, optional): Specific columns to clean. If None, all object columns are cleaned.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    # Remove duplicate rows
    df_clean = df_clean.drop_duplicates()
    
    # Identify columns to clean
    if column_names is None:
        # Clean all object (string) columns
        string_columns = df_clean.select_dtypes(include=['object']).columns
    else:
        # Clean only specified columns
        string_columns = [col for col in column_names if col in df_clean.columns]
    
    # Normalize string columns
    for col in string_columns:
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].apply(
                lambda x: re.sub(r'\s+', ' ', str(x).strip().lower()) if pd.notnull(x) else x
            )
    
    # Reset index after cleaning
    df_clean = df_clean.reset_index(drop=True)
    
    return df_clean

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        email_column (str): Name of the column containing email addresses.
    
    Returns:
        pd.DataFrame: DataFrame with additional validation columns.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    df_valid = df.copy()
    
    # Basic email validation regex
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    df_valid['email_valid'] = df_valid[email_column].apply(
        lambda x: bool(re.match(email_pattern, str(x))) if pd.notnull(x) else False
    )
    
    df_valid['email_domain'] = df_valid[email_column].apply(
        lambda x: str(x).split('@')[-1] if pd.notnull(x) and '@' in str(x) else None
    )
    
    return df_valid

# Example usage function
def demonstrate_cleaning():
    """Demonstrate the data cleaning functions."""
    # Create sample data
    sample_data = {
        'name': ['John Doe', 'Jane Smith', 'john doe', 'Bob Johnson', 'Jane Smith'],
        'email': ['john@example.com', 'jane@test.org', 'invalid-email', 'bob@company.net', 'jane@test.org'],
        'age': [25, 30, 25, 35, 30]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    # Clean the data
    df_clean = clean_dataframe(df)
    print("Cleaned DataFrame:")
    print(df_clean)
    print("\n")
    
    # Validate emails
    df_validated = validate_email_column(df_clean, 'email')
    print("DataFrame with email validation:")
    print(df_validated)
    
    return df_validated

if __name__ == "__main__":
    demonstrate_cleaning()
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas DataFrame column using the IQR method.
    
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

def calculate_summary_stats(data, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame.
    column (str): The column name to analyze.
    
    Returns:
    dict: Dictionary containing count, mean, std, min, and max.
    """
    stats = {
        'count': data[column].count(),
        'mean': data[column].mean(),
        'std': data[column].std(),
        'min': data[column].min(),
        'max': data[column].max()
    }
    return stats