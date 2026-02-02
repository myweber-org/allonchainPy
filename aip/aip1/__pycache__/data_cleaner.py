
import pandas as pd
import re

def clean_dataframe(df, column_name):
    """
    Clean a specific column in a pandas DataFrame by removing duplicates,
    stripping whitespace, and converting to lowercase.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")

    df[column_name] = df[column_name].astype(str)
    df[column_name] = df[column_name].str.strip()
    df[column_name] = df[column_name].str.lower()
    df.drop_duplicates(subset=[column_name], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def normalize_string(text):
    """
    Normalize a string by removing extra spaces and special characters.
    """
    if not isinstance(text, str):
        return text
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def process_file(input_path, output_path, column_to_clean):
    """
    Read a CSV file, clean the specified column, and save to a new file.
    """
    try:
        df = pd.read_csv(input_path)
        df = clean_dataframe(df, column_to_clean)
        df[column_to_clean] = df[column_to_clean].apply(normalize_string)
        df.to_csv(output_path, index=False)
        print(f"Data cleaned and saved to {output_path}")
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    target_column = "product_name"
    process_file(input_file, output_file, target_column)