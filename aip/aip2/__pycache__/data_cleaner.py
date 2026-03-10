import pandas as pd

def clean_data(df, drop_duplicates=True, fill_missing=True):
    """
    Clean the input DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
    fill_missing (bool): Whether to fill missing values. Default is True.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        for column in cleaned_df.columns:
            if cleaned_df[column].dtype in ['int64', 'float64']:
                cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].median())
            else:
                cleaned_df[column] = cleaned_df[column].fillna('Unknown')
    
    return cleaned_df

def main():
    sample_data = {
        'Name': ['Alice', 'Bob', 'Alice', 'Charlie', None],
        'Age': [25, 30, 25, None, 35],
        'Score': [85.5, 90.0, 85.5, 78.0, None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    cleaned = clean_data(df)
    print("Cleaned DataFrame:")
    print(cleaned)

if __name__ == "__main__":
    main()