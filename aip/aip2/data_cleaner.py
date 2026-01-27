
def remove_duplicates_preserve_order(iterable):
    seen = set()
    result = []
    for item in iterable:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import pandas as pd
import numpy as np
from pathlib import Path

def clean_csv_data(input_path, output_path=None, fill_strategy='mean'):
    """
    Clean CSV data by handling missing values and removing duplicates.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path for cleaned CSV file (optional)
        fill_strategy: Strategy for filling missing values ('mean', 'median', 'mode', 'drop')
    
    Returns:
        pandas.DataFrame: Cleaned dataframe
    """
    
    # Read input CSV
    df = pd.read_csv(input_path)
    
    # Remove duplicate rows
    initial_rows = len(df)
    df = df.drop_duplicates()
    duplicates_removed = initial_rows - len(df)
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    
    if fill_strategy == 'drop':
        df = df.dropna()
    elif fill_strategy == 'mean' and len(numeric_cols) > 0:
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif fill_strategy == 'median' and len(numeric_cols) > 0:
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif fill_strategy == 'mode':
        for col in df.columns:
            if col in numeric_cols:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)
            else:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
    
    # Fill remaining categorical columns with 'Unknown'
    for col in categorical_cols:
        df[col] = df[col].fillna('Unknown')
    
    # Save cleaned data if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
    
    # Print cleaning summary
    print(f"Data cleaning completed:")
    print(f"  - Removed {duplicates_removed} duplicate rows")
    print(f"  - Final dataset shape: {df.shape}")
    print(f"  - Missing values filled using: {fill_strategy} strategy")
    
    return df

def validate_dataframe(df, required_columns=None):
    """
    Validate dataframe structure and content.
    
    Args:
        df: pandas.DataFrame to validate
        required_columns: List of required column names
    
    Returns:
        dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Check if dataframe is empty
    if df.empty:
        validation_results['is_valid'] = False
        validation_results['errors'].append('DataFrame is empty')
    
    # Check required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f'Missing required columns: {missing_columns}')
    
    # Check for all-null columns
    null_columns = df.columns[df.isnull().all()].tolist()
    if null_columns:
        validation_results['warnings'].append(f'Columns with all null values: {null_columns}')
    
    # Check data types
    for col in df.columns:
        if df[col].dtype == 'object' and df[col].str.contains('^\s*$').any():
            validation_results['warnings'].append(f"Column '{col}' contains empty strings")
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'name': ['Alice', 'Bob', None, 'David', 'Eve', 'Eve'],
        'age': [25, 30, None, 35, None, 35],
        'score': [85.5, 92.0, 78.5, None, 88.0, 88.0]
    }
    
    # Create sample dataframe
    df = pd.DataFrame(sample_data)
    
    # Save sample data
    df.to_csv('sample_data.csv', index=False)
    
    # Clean the data
    cleaned_df = clean_csv_data('sample_data.csv', 'cleaned_data.csv', fill_strategy='mean')
    
    # Validate cleaned data
    validation = validate_dataframe(cleaned_df, required_columns=['id', 'name', 'age', 'score'])
    
    print(f"\nValidation results:")
    print(f"  Is valid: {validation['is_valid']}")
    print(f"  Errors: {validation['errors']}")
    print(f"  Warnings: {validation['warnings']}")