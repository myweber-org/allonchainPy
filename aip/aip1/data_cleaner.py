
import json

def clean_data(input_file, output_file, key='valid'):
    """
    Load JSON data from input_file, filter entries where the specified key
    is True, and save the cleaned data to output_file.
    """
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            cleaned_data = [entry for entry in data if entry.get(key) is True]
        elif isinstance(data, dict):
            cleaned_data = {k: v for k, v in data.items() if v.get(key) is True}
        else:
            raise ValueError("Unsupported data format. Expected list or dict.")
        
        with open(output_file, 'w') as f:
            json.dump(cleaned_data, f, indent=2)
        
        print(f"Cleaned data saved to {output_file}")
        return len(cleaned_data)
    
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
    except json.JSONDecodeError:
        print(f"Error: File '{input_file}' contains invalid JSON.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")