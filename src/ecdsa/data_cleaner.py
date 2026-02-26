
import pandas as pd
import numpy as np
import sys

def clean_csv(input_file, output_file):
    try:
        df = pd.read_csv(input_file)
        
        print(f"Original shape: {df.shape}")
        print(f"Missing values per column:")
        print(df.isnull().sum())
        
        df_cleaned = df.copy()
        
        for column in df_cleaned.columns:
            if df_cleaned[column].dtype in ['int64', 'float64']:
                df_cleaned[column].fillna(df_cleaned[column].median(), inplace=True)
            elif df_cleaned[column].dtype == 'object':
                df_cleaned[column].fillna(df_cleaned[column].mode()[0] if not df_cleaned[column].mode().empty else 'Unknown', inplace=True)
        
        df_cleaned = df_cleaned.drop_duplicates()
        
        numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
        if not numeric_columns.empty:
            Q1 = df_cleaned[numeric_columns].quantile(0.25)
            Q3 = df_cleaned[numeric_columns].quantile(0.75)
            IQR = Q3 - Q1
            
            outlier_condition = (
                (df_cleaned[numeric_columns] < (Q1 - 1.5 * IQR)) | 
                (df_cleaned[numeric_columns] > (Q3 + 1.5 * IQR))
            )
            
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].mask(
                outlier_condition, 
                df_cleaned[numeric_columns].median()
            )
        
        df_cleaned.to_csv(output_file, index=False)
        
        print(f"Cleaned shape: {df_cleaned.shape}")
        print(f"Cleaned data saved to: {output_file}")
        
        return True
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return False
    except Exception as e:
        print(f"Error during cleaning: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python data_cleaner.py <input_file.csv> <output_file.csv>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    success = clean_csv(input_file, output_file)
    
    if success:
        print("Data cleaning completed successfully.")
    else:
        print("Data cleaning failed.")
        sys.exit(1)