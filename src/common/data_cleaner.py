
def remove_duplicates(data_list):
    """
    Remove duplicate entries from a list while preserving order.
    
    Args:
        data_list: List containing potentially duplicate items.
    
    Returns:
        List with duplicates removed.
    """
    seen = set()
    result = []
    
    for item in data_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    
    return result

def clean_numeric_data(values, remove_negative=False):
    """
    Clean numeric data by removing non-numeric values and optionally negative values.
    
    Args:
        values: List of values to clean.
        remove_negative: Boolean flag to remove negative values.
    
    Returns:
        List of cleaned numeric values.
    """
    cleaned = []
    
    for value in values:
        try:
            num = float(value)
            if remove_negative and num < 0:
                continue
            cleaned.append(num)
        except (ValueError, TypeError):
            continue
    
    return cleaned

if __name__ == "__main__":
    # Example usage
    sample_data = [1, 2, 2, 3, 4, 4, 5, 1, 6]
    cleaned_data = remove_duplicates(sample_data)
    print(f"Original: {sample_data}")
    print(f"Cleaned: {cleaned_data}")
    
    mixed_data = [1, "2", "abc", 3.5, -2, None, 7]
    numeric_data = clean_numeric_data(mixed_data, remove_negative=True)
    print(f"Mixed data: {mixed_data}")
    print(f"Numeric only: {numeric_data}")