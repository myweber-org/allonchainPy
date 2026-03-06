
import csv
import sys

def clean_csv(input_file, output_file):
    seen = set()
    cleaned_rows = []

    try:
        with open(input_file, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            headers = next(reader)
            for row in reader:
                if not row:
                    continue
                row_tuple = tuple(row)
                if row_tuple in seen:
                    continue
                seen.add(row_tuple)
                cleaned_rows.append(row)

        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(headers)
            writer.writerows(cleaned_rows)

        print(f"Cleaned data saved to {output_file}")
        print(f"Removed {len(seen) - len(cleaned_rows)} duplicate rows")

    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python data_cleaner.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    clean_csv(input_file, output_file)