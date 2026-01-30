import pandas as pd
import numpy as np
import sys

def clean_csv(input_file, output_file):
    try:
        df = pd.read_csv(input_file)
        print(f"Original shape: {df.shape}")
        
        # Remove duplicate rows
        df.drop_duplicates(inplace=True)
        print(f"After removing duplicates: {df.shape}")
        
        # Handle missing values
        for column in df.columns:
            if df[column].dtype in ['int64', 'float64']:
                df[column].fillna(df[column].median(), inplace=True)
            else:
                df[column].fillna(df[column].mode()[0] if not df[column].mode().empty else 'Unknown', inplace=True)
        
        # Remove outliers using IQR for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        print(f"After outlier removal: {df.shape}")
        
        # Save cleaned data
        df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to {output_file}")
        return True
        
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
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