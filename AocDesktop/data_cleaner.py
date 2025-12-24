
import pandas as pd
import numpy as np
from pathlib import Path

def clean_csv_data(input_path, output_path=None):
    """
    Load, clean, and save CSV data by handling missing values,
    removing duplicates, and standardizing formats.
    """
    try:
        df = pd.read_csv(input_path)
        
        # Remove duplicate rows
        initial_count = len(df)
        df.drop_duplicates(inplace=True)
        duplicates_removed = initial_count - len(df)
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        df[categorical_cols] = df[categorical_cols].fillna('Unknown')
        
        # Standardize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Remove leading/trailing whitespace from string columns
        for col in categorical_cols:
            df[col] = df[col].astype(str).str.strip()
        
        # Generate output filename if not provided
        if output_path is None:
            input_file = Path(input_path)
            output_path = input_file.parent / f"cleaned_{input_file.name}"
        
        # Save cleaned data
        df.to_csv(output_path, index=False)
        
        # Print summary statistics
        print(f"Data cleaning completed successfully!")
        print(f"Original records: {initial_count}")
        print(f"Duplicates removed: {duplicates_removed}")
        print(f"Final records: {len(df)}")
        print(f"Cleaned data saved to: {output_path}")
        
        return df, output_path
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_path}' not found.")
        return None, None
    except pd.errors.EmptyDataError:
        print(f"Error: Input file '{input_path}' is empty.")
        return None, None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None, None

def validate_dataframe(df):
    """
    Perform basic validation on the cleaned dataframe.
    """
    if df is None or df.empty:
        return False
    
    validation_results = {
        'has_data': not df.empty,
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict()
    }
    
    print("Data Validation Results:")
    for key, value in validation_results.items():
        print(f"  {key}: {value}")
    
    return validation_results['missing_values'] == 0 and validation_results['duplicate_rows'] == 0

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'ID': [1, 2, 3, 4, 5, 5],
        'Name': ['Alice', 'Bob', None, 'David', 'Eve', 'Eve'],
        'Age': [25, 30, None, 35, 40, 40],
        'Score': [85.5, 92.0, 78.5, None, 88.0, 88.0]
    }
    
    # Create sample CSV for demonstration
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        pd.DataFrame(sample_data).to_csv(tmp.name, index=False)
        input_file = tmp.name
    
    print("Processing sample data...")
    cleaned_df, output_file = clean_csv_data(input_file)
    
    if cleaned_df is not None:
        is_valid = validate_dataframe(cleaned_df)
        print(f"Data validation passed: {is_valid}")
    
    # Clean up temporary file
    os.unlink(input_file)