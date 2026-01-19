import pandas as pd
import sys

def remove_duplicates(input_file, output_file=None, subset=None, keep='first'):
    """
    Remove duplicate rows from a CSV file.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file (optional)
        subset (list): Columns to consider for identifying duplicates
        keep (str): Which duplicate to keep - 'first', 'last', or False (remove all)
    
    Returns:
        pandas.DataFrame: Cleaned dataframe
    """
    try:
        df = pd.read_csv(input_file)
        original_rows = len(df)
        
        df_cleaned = df.drop_duplicates(subset=subset, keep=keep)
        removed_rows = original_rows - len(df_cleaned)
        
        if output_file:
            df_cleaned.to_csv(output_file, index=False)
            print(f"Cleaned data saved to: {output_file}")
        
        print(f"Original rows: {original_rows}")
        print(f"Removed duplicates: {removed_rows}")
        print(f"Final rows: {len(df_cleaned)}")
        
        return df_cleaned
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: File '{input_file}' is empty.")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data_cleaner.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    remove_duplicates(input_file, output_file)