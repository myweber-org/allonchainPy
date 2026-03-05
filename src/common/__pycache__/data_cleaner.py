import pandas as pd
import numpy as np

def clean_csv_data(input_file, output_file, strategy='mean'):
    """
    Clean a CSV file by handling missing values.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to save cleaned CSV file
        strategy (str): Strategy for handling missing values.
                       Options: 'mean', 'median', 'mode', 'drop'
    """
    try:
        df = pd.read_csv(input_file)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].isnull().any():
                if strategy == 'mean':
                    fill_value = df[col].mean()
                elif strategy == 'median':
                    fill_value = df[col].median()
                elif strategy == 'mode':
                    fill_value = df[col].mode()[0]
                elif strategy == 'drop':
                    df = df.dropna(subset=[col])
                    continue
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")
                
                df[col] = df[col].fillna(fill_value)
        
        df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to: {output_file}")
        print(f"Original shape: {df.shape}")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except pd.errors.EmptyDataError:
        print("Error: Input file is empty.")
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")

def validate_dataframe(df):
    """
    Validate dataframe for common data quality issues.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
    
    Returns:
        dict: Dictionary containing validation results
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(df.select_dtypes(include=['object']).columns)
    }
    
    return validation_results

if __name__ == "__main__":
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    
    clean_csv_data(input_csv, output_csv, strategy='median')
    
    df_cleaned = pd.read_csv(output_csv)
    validation = validate_dataframe(df_cleaned)
    
    print("\nData Validation Results:")
    for key, value in validation.items():
        print(f"{key}: {value}")