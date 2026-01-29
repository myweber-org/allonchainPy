import pandas as pd

def remove_duplicates(input_file, output_file, key_columns):
    """
    Load data from input_file, remove duplicate rows based on key_columns,
    and save cleaned data to output_file.
    """
    try:
        df = pd.read_csv(input_file)
        initial_count = len(df)
        
        df_cleaned = df.drop_duplicates(subset=key_columns, keep='first')
        final_count = len(df_cleaned)
        
        df_cleaned.to_csv(output_file, index=False)
        
        print(f"Data cleaning completed:")
        print(f"  Initial records: {initial_count}")
        print(f"  Final records: {final_count}")
        print(f"  Duplicates removed: {initial_count - final_count}")
        print(f"  Cleaned data saved to: {output_file}")
        
        return df_cleaned
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_data(df, required_columns):
    """
    Validate that dataframe contains all required columns.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Warning: Missing required columns: {missing_columns}")
        return False
    return True

if __name__ == "__main__":
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    key_fields = ["id", "email", "timestamp"]
    
    cleaned_data = remove_duplicates(input_csv, output_csv, key_fields)
    
    if cleaned_data is not None:
        required_fields = ["id", "name", "email", "value"]
        is_valid = validate_data(cleaned_data, required_fields)
        
        if is_valid:
            print("Data validation passed.")
        else:
            print("Data validation failed.")