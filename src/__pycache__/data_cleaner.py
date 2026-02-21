
import pandas as pd

def clean_dataset(df, sort_column=None):
    """
    Cleans a pandas DataFrame by removing duplicate rows and optionally sorting.
    
    Args:
        df (pd.DataFrame): The input DataFrame to clean.
        sort_column (str, optional): Column name to sort by. Defaults to None.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame with duplicates removed and sorted if specified.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()
    
    # Sort by specified column if provided
    if sort_column and sort_column in df_cleaned.columns:
        df_cleaned = df_cleaned.sort_values(by=sort_column)
    
    # Reset index after cleaning
    df_cleaned = df_cleaned.reset_index(drop=True)
    
    return df_cleaned

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 3, 2, 4, 1],
        'name': ['Alice', 'Bob', 'Charlie', 'Bob', 'David', 'Alice'],
        'score': [85, 92, 78, 92, 88, 85]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame (sorted by 'id'):")
    cleaned_df = clean_dataset(df, sort_column='id')
    print(cleaned_df)