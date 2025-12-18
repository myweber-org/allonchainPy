
import pandas as pd
import numpy as np
from scipy import stats

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Remove outliers using z-score
    z_scores = np.abs(stats.zscore(df[numeric_cols]))
    df = df[(z_scores < 3).all(axis=1)]
    
    # Normalize numeric columns
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
    
    return df

def save_cleaned_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = load_and_clean_data(input_file)
    save_cleaned_data(cleaned_df, output_file)
import pandas as pd
import numpy as np

def clean_dataframe(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop').
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows.")
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
        print("Dropped rows with missing values.")
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
            else:
                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
        print(f"Filled missing numeric values with {fill_missing}.")
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object':
                mode_val = cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else ''
                cleaned_df[col].fillna(mode_val, inplace=True)
        print("Filled missing categorical values with mode.")
    
    return cleaned_df

def validate_dataframe(df, check_nulls=True, check_types=True):
    """
    Validate DataFrame for common data quality issues.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    check_nulls (bool): Check for null values.
    check_types (bool): Check column data types.
    
    Returns:
    dict: Dictionary with validation results.
    """
    validation_results = {}
    
    if check_nulls:
        null_counts = df.isnull().sum()
        validation_results['null_counts'] = null_counts[null_counts > 0].to_dict()
        validation_results['total_nulls'] = null_counts.sum()
    
    if check_types:
        dtypes = df.dtypes.to_dict()
        validation_results['dtypes'] = dtypes
    
    validation_results['shape'] = df.shape
    validation_results['columns'] = list(df.columns)
    
    return validation_results

def main():
    # Example usage
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5, 6],
        'value': [10.5, 20.3, np.nan, 40.1, 50.0, 50.0, np.nan],
        'category': ['A', 'B', 'A', 'B', 'A', 'A', 'C'],
        'score': [100, 200, 150, np.nan, 250, 250, 300]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned_df = clean_dataframe(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    # Validate the cleaned data
    validation = validate_dataframe(cleaned_df)
    print("\nValidation Results:")
    for key, value in validation.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()