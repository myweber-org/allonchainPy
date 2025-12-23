
def remove_duplicates(input_list):
    """
    Remove duplicate elements from a list while preserving order.
    Returns a new list with unique elements.
    """
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def clean_numeric_strings(string_list):
    """
    Clean a list of strings by converting numeric strings to integers.
    Non-numeric strings are kept as is.
    """
    cleaned = []
    for s in string_list:
        s = s.strip()
        if s.isdigit():
            cleaned.append(int(s))
        else:
            cleaned.append(s)
    return cleaned

def filter_by_type(data_list, data_type):
    """
    Filter a list to only include elements of a specific type.
    """
    return [item for item in data_list if isinstance(item, data_type)]

def main():
    # Example usage
    sample_data = [1, 2, 2, 3, "4", "4", "hello", 5.5, 5.5]
    print("Original:", sample_data)
    
    unique_data = remove_duplicates(sample_data)
    print("Unique:", unique_data)
    
    cleaned_data = clean_numeric_strings([str(x) for x in sample_data])
    print("Cleaned numeric strings:", cleaned_data)
    
    ints_only = filter_by_type(sample_data, int)
    print("Integers only:", ints_only)

if __name__ == "__main__":
    main()