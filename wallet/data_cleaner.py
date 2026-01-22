import pandas as pd
import numpy as np

def clean_csv_data(file_path, output_path=None):
    """
    Load a CSV file, perform basic cleaning operations,
    and optionally save the cleaned data.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: File '{file_path}' is empty.")
        return None

    original_shape = df.shape
    print(f"Original data shape: {original_shape}")

    # Remove duplicate rows
    df.drop_duplicates(inplace=True)
    duplicates_removed = original_shape[0] - df.shape[0]
    print(f"Removed {duplicates_removed} duplicate rows.")

    # Fill missing numeric values with column median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"Filled missing values in '{col}' with median: {median_val}")

    # Fill missing categorical values with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
            df[col].fillna(mode_val, inplace=True)
            print(f"Filled missing values in '{col}' with mode: '{mode_val}'")

    # Remove columns with more than 50% missing values
    threshold = 0.5 * len(df)
    cols_to_drop = [col for col in df.columns if df[col].isnull().sum() > threshold]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
        print(f"Dropped columns with >50% missing values: {cols_to_drop}")

    # Reset index after cleaning
    df.reset_index(drop=True, inplace=True)

    final_shape = df.shape
    print(f"Cleaned data shape: {final_shape}")
    print(f"Total rows removed: {original_shape[0] - final_shape[0]}")
    print(f"Total columns removed: {original_shape[1] - final_shape[1]}")

    if output_path:
        try:
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
        except Exception as e:
            print(f"Error saving file: {e}")

    return df

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = clean_csv_data(input_file, output_file)
    if cleaned_df is not None:
        print("Data cleaning completed successfully.")
        print(f"Sample of cleaned data:\n{cleaned_df.head()}")