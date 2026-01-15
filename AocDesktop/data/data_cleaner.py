import pandas as pd
import numpy as np

def clean_csv_data(input_path, output_path, missing_strategy='mean'):
    """
    Clean a CSV file by handling missing values and removing duplicates.
    
    Parameters:
    input_path (str): Path to the input CSV file.
    output_path (str): Path to save the cleaned CSV file.
    missing_strategy (str): Strategy for handling missing values.
                            Options: 'mean', 'median', 'drop', 'zero'.
    """
    try:
        df = pd.read_csv(input_path)
        
        # Remove duplicate rows
        initial_rows = df.shape[0]
        df.drop_duplicates(inplace=True)
        duplicates_removed = initial_rows - df.shape[0]
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if missing_strategy == 'mean':
            for col in numeric_cols:
                df[col].fillna(df[col].mean(), inplace=True)
        elif missing_strategy == 'median':
            for col in numeric_cols:
                df[col].fillna(df[col].median(), inplace=True)
        elif missing_strategy == 'zero':
            df.fillna(0, inplace=True)
        elif missing_strategy == 'drop':
            df.dropna(inplace=True)
        
        # Clean string columns by stripping whitespace
        string_cols = df.select_dtypes(include=['object']).columns
        for col in string_cols:
            df[col] = df[col].str.strip()
        
        # Save cleaned data
        df.to_csv(output_path, index=False)
        
        # Return cleaning statistics
        stats = {
            'original_rows': initial_rows,
            'cleaned_rows': df.shape[0],
            'duplicates_removed': duplicates_removed,
            'columns_cleaned': len(numeric_cols) + len(string_cols)
        }
        
        return stats
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {input_path}")
    except pd.errors.EmptyDataError:
        raise ValueError("The input CSV file is empty")
    except Exception as e:
        raise RuntimeError(f"Error during data cleaning: {str(e)}")

def validate_csv_structure(file_path, required_columns=None):
    """
    Validate the structure of a CSV file.
    
    Parameters:
    file_path (str): Path to the CSV file.
    required_columns (list): List of required column names.
    
    Returns:
    dict: Validation results.
    """
    try:
        df = pd.read_csv(file_path, nrows=1)  # Read only header
        
        validation_result = {
            'file_exists': True,
            'total_columns': len(df.columns),
            'columns': list(df.columns),
            'has_required_columns': True
        }
        
        if required_columns:
            missing_columns = [col for col in required_columns if col not in df.columns]
            validation_result['has_required_columns'] = len(missing_columns) == 0
            validation_result['missing_columns'] = missing_columns
        
        return validation_result
        
    except FileNotFoundError:
        return {'file_exists': False, 'error': 'File not found'}
    except pd.errors.EmptyDataError:
        return {'file_exists': True, 'error': 'File is empty'}

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    # Validate first
    validation = validate_csv_structure(input_file)
    print(f"Validation result: {validation}")
    
    if validation.get('file_exists'):
        try:
            stats = clean_csv_data(input_file, output_file, missing_strategy='mean')
            print(f"Cleaning completed successfully!")
            print(f"Statistics: {stats}")
        except Exception as e:
            print(f"Error during cleaning: {e}")