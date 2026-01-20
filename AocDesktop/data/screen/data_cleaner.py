
import pandas as pd
import numpy as np

def clean_csv_data(input_file, output_file):
    """
    Clean and preprocess CSV data by handling missing values,
    removing duplicates, and standardizing column names.
    """
    try:
        df = pd.read_csv(input_file)
        
        # Standardize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Fill numeric missing values with median
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        
        # Fill categorical missing values with mode
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'unknown')
        
        # Remove outliers using IQR method for numeric columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        # Save cleaned data
        df.to_csv(output_file, index=False)
        print(f"Data cleaning completed. Cleaned data saved to {output_file}")
        return True
        
    except FileNotFoundError:
        print(f"Error: Input file {input_file} not found.")
        return False
    except pd.errors.EmptyDataError:
        print("Error: Input file is empty.")
        return False
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return False

def validate_data(file_path):
    """
    Validate the cleaned data file.
    """
    try:
        df = pd.read_csv(file_path)
        
        # Check for remaining missing values
        missing_values = df.isnull().sum().sum()
        
        # Check data types
        data_types = df.dtypes
        
        # Basic statistics
        stats = df.describe()
        
        print(f"Data validation completed:")
        print(f"Missing values: {missing_values}")
        print(f"Data shape: {df.shape}")
        print(f"Column data types:\n{data_types}")
        
        return {
            'missing_values': missing_values,
            'shape': df.shape,
            'data_types': data_types.to_dict(),
            'stats': stats.to_dict()
        }
        
    except Exception as e:
        print(f"Error during data validation: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    
    # Clean the data
    if clean_csv_data(input_csv, output_csv):
        # Validate the cleaned data
        validation_results = validate_data(output_csv)
        
        if validation_results:
            print("Data cleaning and validation completed successfully.")
        else:
            print("Data validation failed.")