
import pandas as pd
import numpy as np
from pathlib import Path

def clean_csv_data(input_path, output_path=None):
    """
    Clean CSV data by handling missing values and removing duplicates.
    """
    try:
        df = pd.read_csv(input_path)
        
        original_rows = len(df)
        original_columns = len(df.columns)
        
        print(f"Original data: {original_rows} rows, {original_columns} columns")
        
        df_cleaned = df.copy()
        
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        categorical_cols = df_cleaned.select_dtypes(exclude=[np.number]).columns
        
        for col in numeric_cols:
            if df_cleaned[col].isnull().sum() > 0:
                df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
        
        for col in categorical_cols:
            if df_cleaned[col].isnull().sum() > 0:
                df_cleaned[col].fillna(df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else 'Unknown', inplace=True)
        
        df_cleaned.drop_duplicates(inplace=True)
        
        cleaned_rows = len(df_cleaned)
        rows_removed = original_rows - cleaned_rows
        
        print(f"Cleaned data: {cleaned_rows} rows, {original_columns} columns")
        print(f"Rows removed: {rows_removed}")
        
        if output_path is None:
            input_file = Path(input_path)
            output_path = input_file.parent / f"{input_file.stem}_cleaned{input_file.suffix}"
        
        df_cleaned.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
        
        return df_cleaned, output_path
        
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        return None, None
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty")
        return None, None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None, None

def validate_dataframe(df):
    """
    Validate dataframe for common data quality issues.
    """
    if df is None or df.empty:
        print("DataFrame is empty or None")
        return False
    
    issues = []
    
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        issues.append(f"Found {missing_values} missing values")
    
    duplicate_rows = df.duplicated().sum()
    if duplicate_rows > 0:
        issues.append(f"Found {duplicate_rows} duplicate rows")
    
    zero_variance_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if zero_variance_cols:
        issues.append(f"Columns with zero variance: {zero_variance_cols}")
    
    if issues:
        print("Data quality issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    print("Data validation passed")
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 6],
        'name': ['Alice', 'Bob', 'Charlie', None, 'Eve', 'Alice'],
        'age': [25, 30, None, 35, 40, 25],
        'score': [85.5, 92.0, 78.5, None, 88.0, 85.5],
        'department': ['HR', 'IT', 'IT', 'Finance', None, 'HR']
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned_df, output_file = clean_csv_data('test_data.csv')
    
    if cleaned_df is not None:
        validation_result = validate_dataframe(cleaned_df)
        
        print("\nSample of cleaned data:")
        print(cleaned_df.head())
        
        import os
        if os.path.exists('test_data.csv'):
            os.remove('test_data.csv')
        if os.path.exists(output_file):
            os.remove(output_file)