import pandas as pd
import numpy as np

def clean_data(input_file, output_file):
    """
    Load a CSV file, handle missing values by filling with column mean,
    and save the cleaned data to a new CSV file.
    """
    try:
        df = pd.read_csv(input_file)
        print(f"Original data shape: {df.shape}")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                mean_val = df[col].mean()
                df[col].fillna(mean_val, inplace=True)
                print(f"Filled missing values in column '{col}' with mean: {mean_val:.2f}")
        
        df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to: {output_file}")
        print(f"Cleaned data shape: {df.shape}")
        return True
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

if __name__ == "__main__":
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    clean_data(input_csv, output_csv)