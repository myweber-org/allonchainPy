
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
        print(f"  Warnings: {validation['warnings']}")import pandas as pd
import numpy as np

def clean_csv_data(file_path, output_path=None):
    """
    Clean CSV data by handling missing values and converting data types.
    
    Args:
        file_path (str): Path to input CSV file
        output_path (str, optional): Path for cleaned output CSV
    
    Returns:
        pandas.DataFrame: Cleaned DataFrame
    """
    try:
        df = pd.read_csv(file_path)
        
        # Fill missing numeric values with column median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        
        # Fill missing categorical values with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        # Convert date columns if present
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        for col in date_columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                pass
        
        # Remove duplicate rows
        initial_rows = len(df)
        df = df.drop_duplicates()
        duplicates_removed = initial_rows - len(df)
        
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to {output_path}")
        
        print(f"Data cleaning completed:")
        print(f"  - Rows processed: {initial_rows}")
        print(f"  - Duplicates removed: {duplicates_removed}")
        print(f"  - Missing values filled: {df.isnull().sum().sum()}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pandas.DataFrame): DataFrame to validate
        required_columns (list, optional): List of required column names
    
    Returns:
        dict: Validation results
    """
    if df is None or df.empty:
        return {"valid": False, "message": "DataFrame is empty or None"}
    
    validation_results = {
        "valid": True,
        "row_count": len(df),
        "column_count": len(df.columns),
        "missing_values": df.isnull().sum().sum(),
        "duplicate_rows": df.duplicated().sum(),
        "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()}
    }
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results["valid"] = False
            validation_results["missing_columns"] = missing_columns
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 3, 4, 5],
        'value': [10.5, np.nan, 15.2, 20.1, np.nan],
        'category': ['A', 'B', None, 'A', 'C'],
        'date': ['2023-01-01', '2023-01-02', None, '2023-01-04', '2023-01-05']
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned_df = clean_csv_data('test_data.csv', 'cleaned_data.csv')
    
    if cleaned_df is not None:
        validation = validate_dataframe(cleaned_df, required_columns=['id', 'value'])
        print("\nValidation Results:")
        for key, value in validation.items():
            print(f"  {key}: {value}")import pandas as pd
import re

def clean_dataframe(df, text_column):
    """
    Clean a DataFrame by removing duplicates and normalizing text in a specified column.
    """
    # Remove duplicate rows
    df_clean = df.drop_duplicates().reset_index(drop=True)
    
    # Normalize text: lowercase, remove extra whitespace, and strip punctuation
    def normalize_text(text):
        if pd.isna(text):
            return text
        # Convert to lowercase
        text = str(text).lower()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove common punctuation (optional, can be customized)
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    df_clean[text_column] = df_clean[text_column].apply(normalize_text)
    
    return df_clean

def main():
    # Example usage
    data = {
        'id': [1, 2, 3, 4, 2],
        'comment': [
            'Hello World!',
            'hello world',
            '  Hello   World  ',
            'Goodbye',
            'hello world'
        ]
    }
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_dataframe(df, 'comment')
    print("\nCleaned DataFrame:")
    print(cleaned_df)

if __name__ == '__main__':
    main()