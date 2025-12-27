
import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df: pandas DataFrame
        subset: column label or sequence of labels to consider for identifying duplicates
        keep: determines which duplicates to keep ('first', 'last', False)
    
    Returns:
        DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(df) - len(cleaned_df)
    print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df

def clean_missing_values(df, strategy='drop', fill_value=None):
    """
    Handle missing values in a DataFrame.
    
    Args:
        df: pandas DataFrame
        strategy: 'drop' to remove rows with missing values, 
                 'fill' to fill missing values
        fill_value: value to use when filling missing values
    
    Returns:
        DataFrame with missing values handled
    """
    if df.empty:
        return df
    
    if strategy == 'drop':
        cleaned_df = df.dropna()
        removed_count = len(df) - len(cleaned_df)
        print(f"Removed {removed_count} rows with missing values")
    elif strategy == 'fill':
        if fill_value is not None:
            cleaned_df = df.fillna(fill_value)
        else:
            cleaned_df = df.fillna(df.mean(numeric_only=True))
        print("Filled missing values")
    else:
        raise ValueError("Strategy must be 'drop' or 'fill'")
    
    return cleaned_df

def standardize_column_names(df):
    """
    Standardize column names to lowercase with underscores.
    
    Args:
        df: pandas DataFrame
    
    Returns:
        DataFrame with standardized column names
    """
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df

def validate_data(df, required_columns=None):
    """
    Validate data structure and content.
    
    Args:
        df: pandas DataFrame
        required_columns: list of column names that must be present
    
    Returns:
        Boolean indicating if data is valid
    """
    if df.empty:
        print("Warning: DataFrame is empty")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    return True

def main():
    """
    Example usage of data cleaning functions.
    """
    # Create sample data
    data = {
        'Name': ['Alice', 'Bob', 'Alice', 'Charlie', None],
        'Age': [25, 30, 25, 35, 40],
        'City': ['NYC', 'LA', 'NYC', 'Chicago', 'Boston'],
        'Score': [85, 90, 85, 95, None]
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50)
    
    # Clean the data
    df = standardize_column_names(df)
    df = remove_duplicates(df, subset=['name', 'age'])
    df = clean_missing_values(df, strategy='fill', fill_value=0)
    
    print("\nCleaned DataFrame:")
    print(df)
    
    # Validate the cleaned data
    is_valid = validate_data(df, required_columns=['name', 'age', 'city'])
    print(f"\nData validation: {'Passed' if is_valid else 'Failed'}")

if __name__ == "__main__":
    main()
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