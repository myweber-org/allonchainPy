
import csv
import re

def clean_csv(input_file, output_file, remove_duplicates=True, strip_whitespace=True):
    """
    Clean a CSV file by removing duplicates and stripping whitespace.
    
    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to the output cleaned CSV file.
        remove_duplicates (bool): Whether to remove duplicate rows.
        strip_whitespace (bool): Whether to strip whitespace from all fields.
    """
    rows = []
    seen = set()
    
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader)
        rows.append(header)
        
        for row in reader:
            if strip_whitespace:
                row = [field.strip() if isinstance(field, str) else field for field in row]
            
            row_tuple = tuple(row)
            
            if remove_duplicates:
                if row_tuple in seen:
                    continue
                seen.add(row_tuple)
            
            rows.append(row)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(rows)

def validate_email(email):
    """
    Validate an email address using a simple regex pattern.
    
    Args:
        email (str): Email address to validate.
    
    Returns:
        bool: True if email is valid, False otherwise.
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def filter_valid_emails(input_file, output_file, email_column_index):
    """
    Filter rows from a CSV file where the specified column contains a valid email.
    
    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to the output filtered CSV file.
        email_column_index (int): Index of the column containing email addresses.
    """
    valid_rows = []
    
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader)
        valid_rows.append(header)
        
        for row in reader:
            if len(row) > email_column_index:
                email = row[email_column_index].strip()
                if validate_email(email):
                    valid_rows.append(row)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(valid_rows)

if __name__ == "__main__":
    # Example usage
    clean_csv("raw_data.csv", "cleaned_data.csv")
    filter_valid_emails("cleaned_data.csv", "valid_emails.csv", 2)