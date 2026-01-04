
import pandas as pd
import numpy as np

def clean_csv_data(file_path, output_path=None, fill_strategy='mean'):
    """
    Clean CSV data by handling missing values and removing duplicates.
    
    Args:
        file_path (str): Path to input CSV file
        output_path (str, optional): Path for cleaned output CSV. 
                                     If None, returns DataFrame.
        fill_strategy (str): Strategy for filling missing values.
                             Options: 'mean', 'median', 'mode', 'drop'
    
    Returns:
        pd.DataFrame or None: Cleaned DataFrame if output_path is None
    """
    try:
        df = pd.read_csv(file_path)
        
        # Remove duplicate rows
        initial_rows = len(df)
        df = df.drop_duplicates()
        duplicates_removed = initial_rows - len(df)
        
        # Handle missing values
        missing_before = df.isnull().sum().sum()
        
        if fill_strategy == 'drop':
            df = df.dropna()
        elif fill_strategy == 'mean':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].mean())
        elif fill_strategy == 'median':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].median())
        elif fill_strategy == 'mode':
            for col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else None)
        
        missing_after = df.isnull().sum().sum()
        
        # Log cleaning results
        print(f"Data cleaning completed:")
        print(f"  - Duplicates removed: {duplicates_removed}")
        print(f"  - Missing values handled: {missing_before} -> {missing_after}")
        print(f"  - Final rows: {len(df)}")
        
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
            return None
        else:
            return df
            
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    if df.empty:
        validation_results['is_valid'] = False
        validation_results['errors'].append("DataFrame is empty")
        return validation_results
    
    # Check required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing required columns: {missing_columns}")
    
    # Check for all-null columns
    null_columns = df.columns[df.isnull().all()].tolist()
    if null_columns:
        validation_results['warnings'].append(f"Columns with all null values: {null_columns}")
    
    # Check data types
    for col in df.columns:
        if df[col].dtype == 'object':
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.9:
                validation_results['warnings'].append(
                    f"Column '{col}' has high cardinality ({unique_ratio:.1%} unique values)"
                )
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'value': [10.5, None, 15.2, 20.1, None, 10.5],
        'category': ['A', 'B', None, 'A', 'C', 'A']
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('sample_data.csv', index=False)
    
    # Clean the data
    cleaned_df = clean_csv_data('sample_data.csv', fill_strategy='mean')
    
    if cleaned_df is not None:
        # Validate the cleaned data
        validation = validate_dataframe(cleaned_df, required_columns=['id', 'value'])
        print("\nValidation results:")
        print(f"  Valid: {validation['is_valid']}")
        print(f"  Errors: {validation['errors']}")
        print(f"  Warnings: {validation['warnings']}")