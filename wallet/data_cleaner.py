
import pandas as pd
import re

def clean_text_column(df, column_name):
    """
    Standardize text by converting to lowercase, removing extra spaces,
    and stripping special characters.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    df[column_name] = df[column_name].astype(str)
    df[column_name] = df[column_name].str.lower()
    df[column_name] = df[column_name].str.strip()
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'[^\w\s]', '', x))
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'\s+', ' ', x))
    
    return df

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from DataFrame.
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def clean_numeric_column(df, column_name, fill_method='mean'):
    """
    Clean numeric column by filling missing values.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    if fill_method == 'mean':
        fill_value = df[column_name].mean()
    elif fill_method == 'median':
        fill_value = df[column_name].median()
    elif fill_method == 'mode':
        fill_value = df[column_name].mode()[0]
    else:
        fill_value = 0
    
    df[column_name] = df[column_name].fillna(fill_value)
    return df

def validate_email_column(df, column_name):
    """
    Validate email addresses in a column.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    df['is_valid_email'] = df[column_name].str.match(email_pattern)
    return df

def main():
    """
    Example usage of data cleaning functions.
    """
    sample_data = {
        'name': ['John Doe', 'Jane Smith', 'john doe', 'Bob Johnson', 'Jane Smith'],
        'email': ['john@example.com', 'jane@test.org', 'invalid-email', 'bob@company.net', 'jane@test.org'],
        'age': [25, 30, None, 35, 30],
        'notes': ['Hello, World!', '  Extra   Spaces  ', 'special-chars!', 'normal text', '  Extra   Spaces  ']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    df = remove_duplicates(df, subset=['name', 'email'])
    df = clean_text_column(df, 'notes')
    df = clean_numeric_column(df, 'age', fill_method='mean')
    df = validate_email_column(df, 'email')
    
    print("Cleaned DataFrame:")
    print(df)

if __name__ == "__main__":
    main()