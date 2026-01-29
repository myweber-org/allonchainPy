
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def clean_dataset(input_file, output_file):
    try:
        df = pd.read_csv(input_file)
        print(f"Original dataset shape: {df.shape}")
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            original_count = len(df)
            df = remove_outliers_iqr(df, col)
            removed_count = original_count - len(df)
            if removed_count > 0:
                print(f"Removed {removed_count} outliers from column: {col}")
        
        df.to_csv(output_file, index=False)
        print(f"Cleaned dataset saved to: {output_file}")
        print(f"Final dataset shape: {df.shape}")
        
        return True
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return False
    except Exception as e:
        print(f"Error during cleaning: {str(e)}")
        return False

if __name__ == "__main__":
    input_path = "raw_data.csv"
    output_path = "cleaned_data.csv"
    clean_dataset(input_path, output_path)
import re
import pandas as pd
from typing import Optional, List, Dict, Any

def clean_string(text: str) -> str:
    """
    Clean a string by removing extra whitespace and converting to lowercase.
    """
    if not isinstance(text, str):
        return ''
    return re.sub(r'\s+', ' ', text.strip()).lower()

def validate_email(email: str) -> bool:
    """
    Validate an email address format.
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def remove_duplicates(data: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    """
    Remove duplicate dictionaries from a list based on a specified key.
    """
    seen = set()
    unique_data = []
    for item in data:
        value = item.get(key)
        if value not in seen:
            seen.add(value)
            unique_data.append(item)
    return unique_data

def normalize_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Normalize a column in a DataFrame by cleaning each string entry.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    df[column] = df[column].apply(clean_string)
    return df

def filter_valid_emails(emails: List[str]) -> List[str]:
    """
    Filter a list of emails, returning only those with valid format.
    """
    return [email for email in emails if validate_email(email)]