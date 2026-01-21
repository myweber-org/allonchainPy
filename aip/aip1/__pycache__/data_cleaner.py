import pandas as pd
import numpy as np

def clean_csv_data(input_file, output_file):
    """
    Load a CSV file, handle missing values, and save cleaned data.
    """
    try:
        df = pd.read_csv(input_file)
        print(f"Original data shape: {df.shape}")
        
        # Remove duplicate rows
        df.drop_duplicates(inplace=True)
        
        # Fill missing numeric values with column median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Fill missing categorical values with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        # Remove rows where all values are NaN
        df.dropna(how='all', inplace=True)
        
        print(f"Cleaned data shape: {df.shape}")
        print(f"Missing values after cleaning: {df.isnull().sum().sum()}")
        
        df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to: {output_file}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage
    input_path = "raw_data.csv"
    output_path = "cleaned_data.csv"
    cleaned_df = clean_csv_data(input_path, output_path)import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Args:
        data (pd.DataFrame): Input dataframe
        column (str): Column name to clean
    
    Returns:
        pd.DataFrame: Dataframe with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    return filtered_data

def calculate_basic_stats(data, column):
    """
    Calculate basic statistics for a column after outlier removal.
    
    Args:
        data (pd.DataFrame): Input dataframe
        column (str): Column name to analyze
    
    Returns:
        dict: Dictionary containing statistics
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    stats = {
        'mean': np.mean(data[column]),
        'median': np.median(data[column]),
        'std': np.std(data[column]),
        'min': np.min(data[column]),
        'max': np.max(data[column]),
        'count': len(data[column])
    }
    
    return stats

def clean_dataset(data, columns_to_clean):
    """
    Clean multiple columns in a dataset by removing outliers.
    
    Args:
        data (pd.DataFrame): Input dataframe
        columns_to_clean (list): List of column names to clean
    
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    cleaned_data = data.copy()
    
    for column in columns_to_clean:
        if column in cleaned_data.columns:
            cleaned_data = remove_outliers_iqr(cleaned_data, column)
    
    return cleaned_data
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by handling duplicates and missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_missing (str): Strategy to fill missing values ('mean', 'median', 'mode', or 'drop').
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
        print(f"Removed {len(df) - len(cleaned_df)} duplicate rows.")
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
        print("Dropped rows with missing values.")
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        print(f"Filled missing numeric values with {fill_missing}.")
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
        print("Filled missing values with mode.")
    
    print(f"Original shape: {df.shape}, Cleaned shape: {cleaned_df.shape}")
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if df.empty:
        print("Validation failed: DataFrame is empty.")
        return False
    
    if len(df) < min_rows:
        print(f"Validation failed: DataFrame has fewer than {min_rows} rows.")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Validation failed: Missing required columns: {missing_cols}")
            return False
    
    print("Data validation passed.")
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, None],
        'B': [5, None, 7, 8, 9],
        'C': ['x', 'y', 'y', 'z', None]
    }
    df = pd.DataFrame(sample_data)
    
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid = validate_data(cleaned, required_columns=['A', 'B'], min_rows=3)
    print(f"\nData valid: {is_valid}")