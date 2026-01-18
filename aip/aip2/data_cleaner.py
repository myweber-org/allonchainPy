import pandas as pd
import numpy as np
import sys

def clean_csv(input_file, output_file):
    try:
        df = pd.read_csv(input_file)
        
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        df[categorical_cols] = df[categorical_cols].fillna('Unknown')
        
        df.drop_duplicates(inplace=True)
        
        df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to {output_file}")
        return True
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return False
    except pd.errors.EmptyDataError:
        print(f"Error: File '{input_file}' is empty.")
        return False
    except Exception as e:
        print(f"Error during cleaning: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python data_cleaner.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    success = clean_csv(input_file, output_file)
    sys.exit(0 if success else 1)