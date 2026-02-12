
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return resultimport pandas as pd
import numpy as np

def clean_csv_data(file_path, strategy='mean', columns=None):
    """
    Clean a CSV file by handling missing values.
    
    Args:
        file_path (str): Path to the CSV file.
        strategy (str): Strategy for handling missing values.
                        Options: 'mean', 'median', 'mode', 'drop'.
        columns (list): List of column names to apply cleaning to.
                        If None, applies to all numeric columns.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print("Error: File is empty.")
        return None
    
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    for col in columns:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in DataFrame.")
            continue
        
        if df[col].isnull().any():
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mode':
                fill_value = df[col].mode()[0] if not df[col].mode().empty else 0
            elif strategy == 'drop':
                df = df.dropna(subset=[col])
                continue
            else:
                print(f"Warning: Unknown strategy '{strategy}'. Using 'mean'.")
                fill_value = df[col].mean()
            
            df[col].fillna(fill_value, inplace=True)
            print(f"Filled missing values in column '{col}' using {strategy}.")
    
    return df

def save_cleaned_data(df, output_path):
    """
    Save cleaned DataFrame to a CSV file.
    
    Args:
        df (pd.DataFrame): DataFrame to save.
        output_path (str): Path for the output CSV file.
    """
    if df is not None:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to '{output_path}'.")
    else:
        print("Error: No data to save.")

if __name__ == "__main__":
    input_file = "data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = clean_csv_data(input_file, strategy='median')
    save_cleaned_data(cleaned_df, output_file)