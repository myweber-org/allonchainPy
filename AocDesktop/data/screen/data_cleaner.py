
import pandas as pd
import numpy as np
from datetime import datetime

def clean_csv_data(input_path, output_path):
    """
    Clean CSV data by handling missing values, converting data types,
    and removing duplicate rows.
    """
    try:
        df = pd.read_csv(input_path)
        
        print(f"Original shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        df_cleaned = df.copy()
        
        df_cleaned = df_cleaned.drop_duplicates()
        print(f"After removing duplicates: {df_cleaned.shape}")
        
        numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df_cleaned[col].isnull().sum() > 0:
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
        
        categorical_columns = df_cleaned.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if df_cleaned[col].isnull().sum() > 0:
                df_cleaned[col] = df_cleaned[col].fillna('Unknown')
        
        date_columns = []
        for col in df_cleaned.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='coerce')
                    date_columns.append(col)
                except:
                    pass
        
        for col in numeric_columns:
            if df_cleaned[col].dtype in ['int64', 'float64']:
                q1 = df_cleaned[col].quantile(0.25)
                q3 = df_cleaned[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = ((df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)).sum()
                if outliers > 0:
                    df_cleaned[col] = np.where(
                        (df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound),
                        df_cleaned[col].median(),
                        df_cleaned[col]
                    )
                    print(f"Handled {outliers} outliers in column: {col}")
        
        df_cleaned.to_csv(output_path, index=False)
        
        print(f"Cleaned data saved to: {output_path}")
        print(f"Final shape: {df_cleaned.shape}")
        print(f"Missing values after cleaning:")
        print(df_cleaned.isnull().sum())
        
        return df_cleaned
        
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df):
    """
    Validate the cleaned dataframe for common data quality issues.
    """
    if df is None:
        return False
    
    validation_results = {
        'has_duplicates': df.duplicated().sum() == 0,
        'has_nulls': df.isnull().sum().sum() == 0,
        'has_infinite': np.any(np.isinf(df.select_dtypes(include=[np.number]))),
        'row_count': len(df),
        'column_count': len(df.columns)
    }
    
    print("Data Validation Results:")
    for key, value in validation_results.items():
        print(f"{key}: {value}")
    
    return all([validation_results['has_duplicates'], 
                validation_results['has_nulls'] == 0,
                not validation_results['has_infinite']])

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = clean_csv_data(input_file, output_file)
    
    if cleaned_df is not None:
        is_valid = validate_dataframe(cleaned_df)
        if is_valid:
            print("Data cleaning completed successfully. Data is valid.")
        else:
            print("Data cleaning completed with warnings. Check validation results.")