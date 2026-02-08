
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing=True, strategy='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_missing (bool): Whether to fill missing values.
    strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', 'zero').
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows.")
    
    if fill_missing and cleaned_df.isnull().sum().any():
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if cleaned_df[col].isnull().any():
                if strategy == 'mean':
                    fill_value = cleaned_df[col].mean()
                elif strategy == 'median':
                    fill_value = cleaned_df[col].median()
                elif strategy == 'mode':
                    fill_value = cleaned_df[col].mode()[0]
                elif strategy == 'zero':
                    fill_value = 0
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")
                
                missing_count = cleaned_df[col].isnull().sum()
                cleaned_df[col].fillna(fill_value, inplace=True)
                print(f"Filled {missing_count} missing values in column '{col}' with {strategy} value: {fill_value:.4f}")
    
    categorical_cols = cleaned_df.select_dtypes(exclude=[np.number]).columns
    for col in categorical_cols:
        if cleaned_df[col].isnull().any():
            missing_count = cleaned_df[col].isnull().sum()
            cleaned_df[col].fillna('Unknown', inplace=True)
            print(f"Filled {missing_count} missing values in categorical column '{col}' with 'Unknown'")
    
    return cleaned_df

def validate_dataset(df, required_columns=None, min_rows=1):
    """
    Validate dataset structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    tuple: (is_valid, message)
    """
    if len(df) < min_rows:
        return False, f"Dataset must have at least {min_rows} rows, but has {len(df)}"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Dataset is valid"

def get_dataset_summary(df):
    """
    Generate a summary of the dataset.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    
    Returns:
    dict: Summary statistics.
    """
    summary = {
        'rows': len(df),
        'columns': len(df.columns),
        'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(df.select_dtypes(exclude=[np.number]).columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicates': df.duplicated().sum(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
    }
    
    return summary

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'value': [10.5, 20.3, 20.3, None, 40.1, None],
        'category': ['A', 'B', 'B', None, 'A', 'C'],
        'score': [85, 92, 92, 78, None, 88]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    summary = get_dataset_summary(df)
    print("Dataset summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print("\n" + "="*50 + "\n")
    
    cleaned_df = clean_dataset(df, strategy='mean')
    print("\nCleaned dataset:")
    print(cleaned_df)
    
    is_valid, message = validate_dataset(cleaned_df, required_columns=['id', 'value'])
    print(f"\nValidation: {message}")