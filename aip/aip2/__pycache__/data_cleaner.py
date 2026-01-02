
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name to process.
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df.copy()

def calculate_basic_stats(df, column):
    """
    Calculate basic statistics for a DataFrame column.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name to analyze.
    
    Returns:
        dict: Dictionary containing statistical measures.
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

def process_numerical_data(df, columns):
    """
    Process multiple numerical columns by removing outliers and calculating statistics.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list): List of column names to process.
    
    Returns:
        tuple: (cleaned_df, stats_dict)
    """
    cleaned_df = df.copy()
    all_stats = {}
    
    for col in columns:
        if col in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[col]):
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            removed_count = original_count - len(cleaned_df)
            
            stats = calculate_basic_stats(cleaned_df, col)
            stats['outliers_removed'] = removed_count
            all_stats[col] = stats
    
    return cleaned_df, all_stats

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[np.random.choice(df.index, 50), 'A'] = np.random.uniform(300, 500, 50)
    
    print("Original DataFrame shape:", df.shape)
    
    cleaned_df, stats = process_numerical_data(df, ['A', 'B', 'C'])
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("\nStatistics for column 'A':")
    for key, value in stats['A'].items():
        print(f"{key}: {value:.2f}")
def remove_duplicates(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import pandas as pd
import numpy as np
from pathlib import Path

def load_data(filepath):
    """Load data from CSV file."""
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} records from {filepath}")
        return df
    except FileNotFoundError:
        print(f"Error: File {filepath} not found")
        return None

def remove_duplicates(df, subset=None):
    """Remove duplicate rows from DataFrame."""
    if df is None or df.empty:
        return df
    
    initial_count = len(df)
    
    if subset:
        df_clean = df.drop_duplicates(subset=subset, keep='first')
    else:
        df_clean = df.drop_duplicates(keep='first')
    
    removed_count = initial_count - len(df_clean)
    print(f"Removed {removed_count} duplicate records")
    
    return df_clean

def clean_missing_values(df, strategy='drop'):
    """Handle missing values in DataFrame."""
    if df is None or df.empty:
        return df
    
    missing_before = df.isnull().sum().sum()
    
    if strategy == 'drop':
        df_clean = df.dropna()
    elif strategy == 'fill':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_clean = df.copy()
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
    else:
        df_clean = df.copy()
    
    missing_after = df_clean.isnull().sum().sum()
    print(f"Reduced missing values from {missing_before} to {missing_after}")
    
    return df_clean

def validate_data(df):
    """Perform basic data validation."""
    if df is None or df.empty:
        print("No data to validate")
        return False
    
    validation_passed = True
    
    if df.isnull().any().any():
        print("Warning: Data contains missing values")
        validation_passed = False
    
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        print(f"Warning: Data contains {duplicate_count} duplicate rows")
        validation_passed = False
    
    if validation_passed:
        print("Data validation passed")
    
    return validation_passed

def save_clean_data(df, output_path):
    """Save cleaned DataFrame to CSV."""
    if df is None or df.empty:
        print("No data to save")
        return False
    
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} records to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving data: {e}")
        return False

def main():
    """Main function to execute data cleaning pipeline."""
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    print("Starting data cleaning process...")
    
    df = load_data(input_file)
    if df is None:
        return
    
    print("Removing duplicates...")
    df = remove_duplicates(df, subset=['id', 'timestamp'])
    
    print("Cleaning missing values...")
    df = clean_missing_values(df, strategy='fill')
    
    print("Validating data...")
    if validate_data(df):
        print("Data cleaning completed successfully")
    else:
        print("Data cleaning completed with warnings")
    
    save_clean_data(df, output_file)
    
    print("Data cleaning pipeline finished")

if __name__ == "__main__":
    main()