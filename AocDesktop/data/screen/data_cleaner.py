import pandas as pd
import re

def clean_text_column(df, column_name):
    """
    Standardize text in a DataFrame column by converting to lowercase,
    removing extra whitespace, and stripping special characters.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    df[column_name] = df[column_name].astype(str).str.lower()
    df[column_name] = df[column_name].str.strip()
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'[^\w\s]', '', x))
    df[column_name] = df[column_name].str.replace(r'\s+', ' ', regex=True)
    
    return df

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    """
    return df.drop_duplicates(subset=subset, keep=keep)

def validate_email_column(df, column_name):
    """
    Validate email addresses in a DataFrame column.
    Returns a boolean Series indicating valid emails.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return df[column_name].str.match(email_pattern)

def main():
    # Example usage
    data = {
        'name': ['John Doe', 'Jane Smith', 'john doe', 'Bob Johnson  ', 'ALICE'],
        'email': ['john@example.com', 'jane@test.org', 'invalid-email', 'bob@company.com', 'alice@domain.net'],
        'age': [25, 30, 25, 35, 28]
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print()
    
    # Clean text column
    df_cleaned = clean_text_column(df.copy(), 'name')
    print("After cleaning 'name' column:")
    print(df_cleaned)
    print()
    
    # Remove duplicates
    df_no_dupes = remove_duplicates(df_cleaned, subset=['name'], keep='first')
    print("After removing duplicate names:")
    print(df_no_dupes)
    print()
    
    # Validate emails
    valid_emails = validate_email_column(df, 'email')
    print("Valid email addresses:")
    print(valid_emails)

if __name__ == "__main__":
    main()