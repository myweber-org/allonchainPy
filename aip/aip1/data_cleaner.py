import csv
import re

def clean_csv(input_file, output_file, columns_to_clean=None):
    """
    Clean a CSV file by removing extra whitespace and optionally cleaning specific columns.
    
    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to the output cleaned CSV file.
        columns_to_clean (list, optional): List of column names to apply cleaning to.
                                          If None, all columns are cleaned.
    """
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for row in reader:
                cleaned_row = {}
                for field in fieldnames:
                    value = row[field]
                    if value is not None and (columns_to_clean is None or field in columns_to_clean):
                        # Remove extra whitespace
                        value = re.sub(r'\s+', ' ', str(value)).strip()
                    cleaned_row[field] = value
                writer.writerow(cleaned_row)

def validate_email(email):
    """
    Validate an email address format.
    
    Args:
        email (str): Email address to validate.
    
    Returns:
        bool: True if email format is valid, False otherwise.
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def remove_duplicates(input_file, output_file, key_column):
    """
    Remove duplicate rows from a CSV file based on a key column.
    
    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to the output CSV file without duplicates.
        key_column (str): Column name to use for identifying duplicates.
    """
    seen = set()
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for row in reader:
                key = row[key_column]
                if key not in seen:
                    seen.add(key)
                    writer.writerow(row)

if __name__ == "__main__":
    # Example usage
    clean_csv('input.csv', 'cleaned.csv', columns_to_clean=['name', 'email'])
    remove_duplicates('cleaned.csv', 'final.csv', key_column='id')
    
    # Test email validation
    test_emails = ['test@example.com', 'invalid-email', 'another.test@domain.co.uk']
    for email in test_emails:
        print(f"{email}: {validate_email(email)}")