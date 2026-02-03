import csv
import re

def clean_csv(input_file, output_file):
    """
    Clean a CSV file by removing rows with missing values,
    stripping whitespace, and standardizing date formats.
    """
    cleaned_rows = []
    
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        
        for row in reader:
            # Skip rows with any empty values
            if any(value.strip() == '' for value in row.values()):
                continue
            
            # Strip whitespace from all fields
            cleaned_row = {key: value.strip() for key, value in row.items()}
            
            # Standardize date format (MM/DD/YYYY to YYYY-MM-DD)
            date_pattern = r'(\d{1,2})/(\d{1,2})/(\d{4})'
            for key in cleaned_row:
                match = re.match(date_pattern, cleaned_row[key])
                if match:
                    month, day, year = match.groups()
                    cleaned_row[key] = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            
            cleaned_rows.append(cleaned_row)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned_rows)
    
    return len(cleaned_rows)

if __name__ == "__main__":
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    
    try:
        processed_count = clean_csv(input_csv, output_csv)
        print(f"Successfully processed {processed_count} rows.")
        print(f"Cleaned data saved to {output_csv}")
    except FileNotFoundError:
        print(f"Error: Input file '{input_csv}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")