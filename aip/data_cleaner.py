
import pandas as pd
import numpy as np
import re

def clean_csv_data(input_file, output_file):
    """
    Clean and preprocess CSV data by handling missing values,
    standardizing formats, and removing duplicates.
    """
    try:
        df = pd.read_csv(input_file)
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        # Standardize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        df[categorical_cols] = df[categorical_cols].fillna('unknown')
        
        # Clean string columns
        for col in categorical_cols:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].apply(lambda x: re.sub(r'\s+', ' ', x))
        
        # Remove rows with invalid numeric values
        for col in numeric_cols:
            df = df[~df[col].isin([np.inf, -np.inf])]
        
        # Save cleaned data
        df.to_csv(output_file, index=False)
        print(f"Data cleaning complete. Cleaned data saved to {output_file}")
        return True
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return False
    except pd.errors.EmptyDataError:
        print("Error: Input file is empty.")
        return False
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return False

if __name__ == "__main__":
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    clean_csv_data(input_csv, output_csv)