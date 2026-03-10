
import pandas as pd
import re

def clean_dataframe(df, column_names):
    """
    Clean a pandas DataFrame by removing duplicate rows and normalizing
    specified string columns (strip whitespace, lowercase).
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates().reset_index(drop=True)
    
    # Normalize specified string columns
    for col in column_names:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].astype(str).apply(
                lambda x: re.sub(r'\s+', ' ', x.strip().lower())
            )
    
    return df_cleaned

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column and add a validation flag.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    df['email_valid'] = df[email_column].str.match(email_pattern, na=False)
    
    return df

def save_cleaned_data(df, output_path, format='csv'):
    """
    Save cleaned DataFrame to a file in specified format.
    """
    if format == 'csv':
        df.to_csv(output_path, index=False)
    elif format == 'excel':
        df.to_excel(output_path, index=False)
    elif format == 'json':
        df.to_json(output_path, orient='records')
    else:
        raise ValueError("Unsupported format. Choose 'csv', 'excel', or 'json'.")
    
    print(f"Data successfully saved to {output_path}")