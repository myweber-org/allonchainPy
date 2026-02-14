import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    if max_val - min_val == 0:
        return df[column].apply(lambda x: 0.0)
    return (df[column] - min_val) / (max_val - min_val)

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
    return cleaned_df.reset_index(drop=True)

def validate_dataframe(df):
    required_checks = [
        (lambda x: isinstance(x, pd.DataFrame), "Input must be a pandas DataFrame"),
        (lambda x: not x.empty, "DataFrame cannot be empty"),
        (lambda x: x.isnull().sum().sum() == 0, "DataFrame contains null values")
    ]
    for check, message in required_checks:
        if not check(df):
            raise ValueError(message)
    return True

if __name__ == "__main__":
    sample_data = {'values': [10, 12, 12, 13, 12, 11, 10, 100, 12, 14, 12, 10]}
    df = pd.DataFrame(sample_data)
    print("Original data:")
    print(df)
    
    try:
        validate_dataframe(df)
        cleaned = clean_dataset(df, ['values'])
        print("\nCleaned data:")
        print(cleaned)
    except ValueError as e:
        print(f"Validation error: {e}")
import pandas as pd
import numpy as np
import sys

def clean_csv_data(input_file, output_file):
    """
    Clean CSV data by handling missing values and removing duplicates.
    """
    try:
        df = pd.read_csv(input_file)
        
        # Remove duplicate rows
        df_cleaned = df.drop_duplicates()
        
        # Fill missing numeric values with column median
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
        
        # Fill missing categorical values with mode
        categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else 'Unknown')
        
        # Save cleaned data
        df_cleaned.to_csv(output_file, index=False)
        print(f"Data cleaning completed. Cleaned data saved to {output_file}")
        print(f"Original rows: {len(df)}, Cleaned rows: {len(df_cleaned)}")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python data_cleaner.py <input_file.csv> <output_file.csv>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    clean_csv_data(input_file, output_file)