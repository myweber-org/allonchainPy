import pandas as pd
import sys

def remove_duplicates(input_file, output_file=None):
    """
    Remove duplicate rows from a CSV file.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str, optional): Path to output CSV file. 
                                    If None, overwrites input file.
    """
    try:
        df = pd.read_csv(input_file)
        initial_rows = len(df)
        df_clean = df.drop_duplicates()
        final_rows = len(df_clean)
        
        if output_file is None:
            output_file = input_file
            
        df_clean.to_csv(output_file, index=False)
        
        duplicates_removed = initial_rows - final_rows
        print(f"Successfully removed {duplicates_removed} duplicate rows.")
        print(f"Original rows: {initial_rows}, Cleaned rows: {final_rows}")
        
        return df_clean
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: File '{input_file}' is empty.")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data_cleaner.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    remove_duplicates(input_file, output_file)
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(dataframe, columns):
    cleaned_df = dataframe.copy()
    for col in columns:
        Q1 = cleaned_df[col].quantile(0.25)
        Q3 = cleaned_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
    return cleaned_df

def normalize_data(dataframe, columns, method='minmax'):
    normalized_df = dataframe.copy()
    for col in columns:
        if method == 'minmax':
            min_val = normalized_df[col].min()
            max_val = normalized_df[col].max()
            normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
        elif method == 'zscore':
            mean_val = normalized_df[col].mean()
            std_val = normalized_df[col].std()
            normalized_df[col] = (normalized_df[col] - mean_val) / std_val
    return normalized_df

def handle_missing_values(dataframe, columns, strategy='mean'):
    processed_df = dataframe.copy()
    for col in columns:
        if strategy == 'mean':
            fill_value = processed_df[col].mean()
        elif strategy == 'median':
            fill_value = processed_df[col].median()
        elif strategy == 'mode':
            fill_value = processed_df[col].mode()[0]
        else:
            fill_value = 0
        processed_df[col].fillna(fill_value, inplace=True)
    return processed_df

def clean_dataset(dataframe, numeric_columns):
    df_no_missing = handle_missing_values(dataframe, numeric_columns, 'median')
    df_no_outliers = remove_outliers_iqr(df_no_missing, numeric_columns)
    df_normalized = normalize_data(df_no_outliers, numeric_columns, 'zscore')
    return df_normalized
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        multiplier: IQR multiplier for outlier detection
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def normalize_minmax(data, column):
    """
    Normalize data using min-max scaling to range [0, 1].
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        Series with normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return pd.Series([0.5] * len(data), index=data.index)
    
    return (data[column] - min_val) / (max_val - min_val)

def standardize_zscore(data, column):
    """
    Standardize data using z-score normalization.
    
    Args:
        data: pandas DataFrame
        column: column name to standardize
    
    Returns:
        Series with standardized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    
    return (data[column] - mean_val) / std_val

def clean_dataset(df, numeric_columns=None, outlier_multiplier=1.5):
    """
    Clean dataset by removing outliers and normalizing numeric columns.
    
    Args:
        df: pandas DataFrame to clean
        numeric_columns: list of numeric column names to process
        outlier_multiplier: IQR multiplier for outlier detection
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            # Remove outliers
            q1 = cleaned_df[col].quantile(0.25)
            q3 = cleaned_df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - outlier_multiplier * iqr
            upper_bound = q3 + outlier_multiplier * iqr
            
            mask = (cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)
            cleaned_df = cleaned_df[mask]
            
            # Normalize
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
    
    return cleaned_df.reset_index(drop=True)

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: list of required column names
        min_rows: minimum number of rows required
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'feature1': [1, 2, 3, 4, 100, 6, 7, 8, 9, 10],
        'feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    # Remove outliers from feature1
    cleaned = remove_outliers_iqr(df, 'feature1')
    print("\nDataFrame after removing outliers from feature1:")
    print(cleaned)
    
    # Normalize feature2
    normalized = normalize_minmax(df, 'feature2')
    print("\nNormalized feature2:")
    print(normalized)
    
    # Clean entire dataset
    cleaned_dataset = clean_dataset(df, ['feature1', 'feature2'])
    print("\nCleaned dataset:")
    print(cleaned_dataset)
    
    # Validate DataFrame
    is_valid, message = validate_dataframe(df, ['feature1', 'feature2', 'category'])
    print(f"\nValidation result: {is_valid}, Message: {message}")
import pandas as pd

def clean_dataset(df, column_to_sort=None, keep='first'):
    """
    Clean a pandas DataFrame by removing duplicate rows and optionally sorting by a column.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        column_to_sort (str, optional): Column name to sort by. Defaults to None.
        keep (str, optional): Which duplicates to keep. 'first', 'last', or False. Defaults to 'first'.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame with duplicates removed and optionally sorted.
    """
    cleaned_df = df.drop_duplicates(keep=keep)
    
    if column_to_sort and column_to_sort in cleaned_df.columns:
        cleaned_df = cleaned_df.sort_values(by=column_to_sort).reset_index(drop=True)
    
    return cleaned_df

def main():
    sample_data = {
        'id': [1, 2, 2, 3, 4, 4, 5],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', 'David', 'David', 'Eve'],
        'score': [85, 90, 90, 78, 92, 92, 88]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    cleaned = clean_dataset(df, column_to_sort='score', keep='first')
    print("Cleaned DataFrame (sorted by score):")
    print(cleaned)

if __name__ == "__main__":
    main()