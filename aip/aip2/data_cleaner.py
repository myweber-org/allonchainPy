
import pandas as pd
import numpy as np

def clean_dataset(df, text_columns=None, drop_threshold=0.5):
    """
    Clean dataset by handling missing values and standardizing text columns.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    text_columns (list): List of text column names to standardize
    drop_threshold (float): Threshold for dropping columns with too many nulls
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    
    # Create a copy to avoid modifying original
    cleaned_df = df.copy()
    
    # Drop columns with too many null values
    null_percentages = cleaned_df.isnull().mean()
    columns_to_drop = null_percentages[null_percentages > drop_threshold].index
    cleaned_df = cleaned_df.drop(columns=columns_to_drop)
    
    # Fill numeric columns with median
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    
    # Fill categorical/text columns with mode
    categorical_cols = cleaned_df.select_dtypes(exclude=[np.number]).columns
    for col in categorical_cols:
        cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 'Unknown')
    
    # Standardize text columns if specified
    if text_columns:
        for col in text_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].astype(str).str.lower().str.strip()
    
    # Remove duplicate rows
    cleaned_df = cleaned_df.drop_duplicates()
    
    # Reset index
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_cleaning(original_df, cleaned_df):
    """
    Validate the cleaning process by comparing original and cleaned dataframes.
    
    Parameters:
    original_df (pd.DataFrame): Original dataframe
    cleaned_df (pd.DataFrame): Cleaned dataframe
    
    Returns:
    dict: Dictionary with validation metrics
    """
    validation_results = {
        'original_rows': len(original_df),
        'cleaned_rows': len(cleaned_df),
        'rows_removed': len(original_df) - len(cleaned_df),
        'original_columns': len(original_df.columns),
        'cleaned_columns': len(cleaned_df.columns),
        'columns_removed': len(original_df.columns) - len(cleaned_df.columns),
        'original_null_count': original_df.isnull().sum().sum(),
        'cleaned_null_count': cleaned_df.isnull().sum().sum(),
        'original_duplicates': original_df.duplicated().sum(),
        'cleaned_duplicates': cleaned_df.duplicated().sum()
    }
    
    return validation_results

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'id': [1, 2, 3, 4, 5, 6],
        'name': ['John', 'Jane', None, 'Bob', 'Alice', 'Alice'],
        'age': [25, 30, None, 35, None, 35],
        'score': [85.5, 92.0, 78.5, None, 88.0, 88.0],
        'city': ['New York', 'Los Angeles', 'Chicago', None, 'Boston', 'Boston']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned = clean_dataset(df, text_columns=['name', 'city'])
    print("Cleaned DataFrame:")
    print(cleaned)
    print("\n" + "="*50 + "\n")
    
    # Validate cleaning
    validation = validate_cleaning(df, cleaned)
    print("Validation Results:")
    for key, value in validation.items():
        print(f"{key}: {value}")