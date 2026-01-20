import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        subset (list, optional): Column labels to consider for duplicates.
        keep (str, optional): Which duplicates to keep.
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    return cleaned_df

def clean_numeric_columns(df, columns):
    """
    Clean numeric columns by removing non-numeric characters.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list): List of column names to clean.
    
    Returns:
        pd.DataFrame: DataFrame with cleaned numeric columns.
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def validate_dataframe(df, required_columns):
    """
    Validate if DataFrame contains required columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        required_columns (list): List of required column names.
    
    Returns:
        bool: True if all required columns are present.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        return False
    
    return True

def main():
    # Example usage
    data = {
        'id': [1, 2, 2, 3, 4],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', 'David'],
        'age': ['25', '30', '30', '35', '40'],
        'score': ['95.5', '88.0', '88.0', '92.3', '85.7']
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    
    # Remove duplicates
    df_clean = remove_duplicates(df, subset=['id', 'name'])
    print("\nDataFrame after removing duplicates:")
    print(df_clean)
    
    # Clean numeric columns
    df_clean = clean_numeric_columns(df_clean, ['age', 'score'])
    print("\nDataFrame after cleaning numeric columns:")
    print(df_clean)
    
    # Validate required columns
    required = ['id', 'name', 'age']
    is_valid = validate_dataframe(df_clean, required)
    print(f"\nDataFrame validation: {is_valid}")

if __name__ == "__main__":
    main()