import pandas as pd
import re

def clean_dataframe(df, column_names):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing specified string columns.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates().reset_index(drop=True)

    for col in column_names:
        if col in df_cleaned.columns and df_cleaned[col].dtype == 'object':
            # Convert to string, strip whitespace, and convert to lowercase
            df_cleaned[col] = df_cleaned[col].astype(str).str.strip().str.lower()
            # Remove extra spaces
            df_cleaned[col] = df_cleaned[col].apply(lambda x: re.sub(r'\s+', ' ', x))

    return df_cleaned

def validate_email(email_series):
    """
    Validate email addresses in a pandas Series.
    Returns a boolean Series indicating valid emails.
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return email_series.str.match(pattern, na=False)

# Example usage (commented out)
# if __name__ == "__main__":
#     data = {'Name': ['Alice', 'Bob  ', 'Alice', '  charlie  '],
#             'Email': ['alice@example.com', 'invalid-email', 'alice@example.com', 'charlie@test.co.uk']}
#     df = pd.DataFrame(data)
#     cleaned_df = clean_dataframe(df, ['Name'])
#     print(cleaned_df)
#     print(validate_email(cleaned_df['Email']))