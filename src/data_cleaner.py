
import pandas as pd
import numpy as np

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

def calculate_summary_statistics(df, column):
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
        'count': df[column].count()
    }
    
    return stats

def process_numerical_data(df, columns=None):
    """
    Process numerical columns by removing outliers and calculating statistics.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to process. If None, process all numerical columns.
    
    Returns:
    tuple: (cleaned_df, statistics_dict)
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    statistics = {}
    
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            removed_count = original_count - len(cleaned_df)
            
            stats = calculate_summary_statistics(cleaned_df, col)
            stats['outliers_removed'] = removed_count
            
            statistics[col] = stats
    
    return cleaned_df, statistics

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[::100, 'A'] = 500
    
    print("Original DataFrame shape:", df.shape)
    print("\nOriginal summary:")
    print(df.describe())
    
    cleaned_df, stats = process_numerical_data(df, ['A', 'B'])
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("\nCleaning statistics:")
    for col, col_stats in stats.items():
        print(f"\n{col}:")
        for stat_name, stat_value in col_stats.items():
            print(f"  {stat_name}: {stat_value:.2f}")import pandas as pd
import numpy as np

def clean_csv_data(input_path, output_path, missing_strategy='mean'):
    """
    Clean CSV data by handling missing values and removing duplicates.
    
    Args:
        input_path (str): Path to input CSV file
        output_path (str): Path to save cleaned CSV file
        missing_strategy (str): Strategy for handling missing values
                               ('mean', 'median', 'drop', 'zero')
    
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    
    try:
        df = pd.read_csv(input_path)
        
        print(f"Original data shape: {df.shape}")
        print(f"Missing values per column:")
        print(df.isnull().sum())
        
        df_cleaned = df.copy()
        
        if missing_strategy == 'drop':
            df_cleaned = df_cleaned.dropna()
        elif missing_strategy in ['mean', 'median']:
            numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if missing_strategy == 'mean':
                    fill_value = df_cleaned[col].mean()
                else:
                    fill_value = df_cleaned[col].median()
                
                df_cleaned[col] = df_cleaned[col].fillna(fill_value)
        elif missing_strategy == 'zero':
            df_cleaned = df_cleaned.fillna(0)
        
        df_cleaned = df_cleaned.drop_duplicates()
        
        df_cleaned.to_csv(output_path, index=False)
        
        print(f"Cleaned data shape: {df_cleaned.shape}")
        print(f"Cleaned data saved to: {output_path}")
        
        return df_cleaned
        
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate dataframe structure and content.
    
    Args:
        df (pd.DataFrame): Dataframe to validate
        required_columns (list): List of required column names
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    
    if df is None or df.empty:
        print("Validation failed: Dataframe is empty or None")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            print(f"Validation failed: Missing required columns: {missing_cols}")
            return False
    
    if df.isnull().sum().sum() > 0:
        print("Warning: Dataframe contains missing values")
    
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5],
        'value': [10.5, None, 15.2, None, 20.1],
        'category': ['A', 'B', 'A', 'B', 'A']
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_input.csv', index=False)
    
    cleaned_df = clean_csv_data('test_input.csv', 'test_output.csv', 'mean')
    
    if cleaned_df is not None:
        is_valid = validate_dataframe(cleaned_df, ['id', 'value', 'category'])
        print(f"Data validation result: {is_valid}")
import csv
import re

def clean_csv(input_file, output_file):
    """
    Clean a CSV file by removing rows with missing values,
    stripping whitespace, and converting text to lowercase.
    """
    cleaned_rows = []
    
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader)
        cleaned_rows.append(header)
        
        for row in reader:
            # Skip rows with missing values
            if any(cell.strip() == '' for cell in row):
                continue
            
            # Clean each cell
            cleaned_row = []
            for cell in row:
                # Strip whitespace
                cell = cell.strip()
                # Convert to lowercase
                cell = cell.lower()
                # Remove extra spaces
                cell = re.sub(r'\s+', ' ', cell)
                cleaned_row.append(cell)
            
            cleaned_rows.append(cleaned_row)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(cleaned_rows)
    
    return len(cleaned_rows) - 1  # Return number of cleaned rows (excluding header)

def validate_email(email):
    """
    Validate email format using regex.
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def remove_duplicates(input_file, output_file, key_column=0):
    """
    Remove duplicate rows based on a specified column.
    """
    seen = set()
    unique_rows = []
    
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader)
        unique_rows.append(header)
        
        for row in reader:
            key = row[key_column]
            if key not in seen:
                seen.add(key)
                unique_rows.append(row)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(unique_rows)
    
    return len(unique_rows) - 1