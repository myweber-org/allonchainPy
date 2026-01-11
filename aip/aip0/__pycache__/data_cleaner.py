
import re

def clean_text(text):
    """
    Clean and normalize a given text string.
    
    Args:
        text (str): The input text to be cleaned.
    
    Returns:
        str: The cleaned text with extra whitespace removed and converted to lowercase.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string")
    
    # Remove leading and trailing whitespace
    text = text.strip()
    
    # Replace multiple spaces or newlines with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Convert to lowercase
    text = text.lower()
    
    return text

def clean_text_list(text_list):
    """
    Clean a list of text strings.
    
    Args:
        text_list (list): A list of text strings to be cleaned.
    
    Returns:
        list: A list of cleaned text strings.
    """
    if not isinstance(text_list, list):
        raise TypeError("Input must be a list")
    
    return [clean_text(text) for text in text_list]
import pandas as pd
import numpy as np

def clean_dataset(df, columns_to_check=None, fill_missing='mean', remove_duplicates=True):
    """
    Clean a pandas DataFrame by handling missing values and removing duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns_to_check (list): List of columns to check for missing values
    fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', 'drop')
    remove_duplicates (bool): Whether to remove duplicate rows
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    
    df_clean = df.copy()
    
    if columns_to_check is None:
        columns_to_check = df_clean.columns.tolist()
    
    if remove_duplicates:
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed = initial_rows - len(df_clean)
        print(f"Removed {removed} duplicate rows")
    
    missing_counts = df_clean[columns_to_check].isnull().sum()
    if missing_counts.sum() > 0:
        print("Missing values found:")
        for col, count in missing_counts.items():
            if count > 0:
                print(f"  {col}: {count} missing values")
        
        if fill_missing == 'drop':
            df_clean = df_clean.dropna(subset=columns_to_check)
            print("Rows with missing values dropped")
        else:
            for col in columns_to_check:
                if df_clean[col].isnull().sum() > 0:
                    if fill_missing == 'mean' and pd.api.types.is_numeric_dtype(df_clean[col]):
                        fill_value = df_clean[col].mean()
                    elif fill_missing == 'median' and pd.api.types.is_numeric_dtype(df_clean[col]):
                        fill_value = df_clean[col].median()
                    elif fill_missing == 'mode':
                        fill_value = df_clean[col].mode()[0]
                    else:
                        fill_value = 0 if pd.api.types.is_numeric_dtype(df_clean[col]) else 'Unknown'
                    
                    df_clean[col] = df_clean[col].fillna(fill_value)
                    print(f"Filled missing values in {col} with {fill_value}")
    
    return df_clean

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame")
        return False
    
    if len(df) < min_rows:
        print(f"Error: DataFrame has less than {min_rows} rows")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    return True

def remove_outliers_iqr(df, columns, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of columns to check for outliers
    multiplier (float): IQR multiplier for outlier detection
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    
    df_clean = df.copy()
    
    for col in columns:
        if pd.api.types.is_numeric_dtype(df_clean[col]):
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            initial_count = len(df_clean)
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
            removed = initial_count - len(df_clean)
            
            if removed > 0:
                print(f"Removed {removed} outliers from column '{col}'")
    
    return df_clean

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5, 6],
        'name': ['Alice', 'Bob', 'Charlie', None, 'Eve', 'Eve', 'Frank'],
        'age': [25, 30, None, 40, 35, 35, 150],
        'score': [85.5, 92.0, 78.5, None, 88.0, 88.0, 95.5]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned_df = clean_dataset(df, fill_missing='mean', remove_duplicates=True)
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    is_valid = validate_dataframe(cleaned_df, required_columns=['id', 'name', 'age', 'score'])
    print(f"\nDataFrame validation: {is_valid}")
    
    df_no_outliers = remove_outliers_iqr(cleaned_df, ['age', 'score'])
    print("\nDataFrame after outlier removal:")
    print(df_no_outliers)import pandas as pd
import numpy as np

def clean_csv_data(input_path, output_path):
    """
    Load a CSV file, clean the data by handling missing values,
    converting data types, and save the cleaned version.
    """
    try:
        df = pd.read_csv(input_path)
        
        # Fill missing numeric values with column median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        
        # Fill missing categorical values with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        # Convert date columns if present
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        for col in date_columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                pass
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        # Reset index after cleaning
        df = df.reset_index(drop=True)
        
        # Save cleaned data
        df.to_csv(output_path, index=False)
        print(f"Data cleaning completed. Cleaned data saved to: {output_path}")
        return df
        
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df):
    """
    Validate the cleaned dataframe for basic quality checks.
    """
    if df is None or df.empty:
        print("DataFrame is empty or None")
        return False
    
    checks = {
        'has_nulls': df.isnull().sum().sum() == 0,
        'has_duplicates': not df.duplicated().any(),
        'has_rows': len(df) > 0,
        'has_columns': len(df.columns) > 0
    }
    
    all_passed = all(checks.values())
    
    if all_passed:
        print("All data validation checks passed")
    else:
        failed = [k for k, v in checks.items() if not v]
        print(f"Validation failed for: {', '.join(failed)}")
    
    return all_passed

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = clean_csv_data(input_file, output_file)
    
    if cleaned_df is not None:
        validate_dataframe(cleaned_df)
import pandas as pd
import numpy as np
from scipy import stats

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    for column in df.select_dtypes(include=[np.number]).columns:
        df[column] = df[column].fillna(df[column].median())
    
    # Remove outliers using z-score
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    z_scores = np.abs(stats.zscore(df[numeric_cols]))
    df = df[(z_scores < 3).all(axis=1)]
    
    # Normalize numeric columns
    for column in numeric_cols:
        if df[column].std() > 0:
            df[column] = (df[column] - df[column].mean()) / df[column].std()
    
    return df

def save_cleaned_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = load_and_clean_data(input_file)
    save_cleaned_data(cleaned_df, output_file)