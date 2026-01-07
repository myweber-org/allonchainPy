
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

def clean_numeric_data(values, default=0):
    """
    Clean numeric data by converting strings to floats and handling invalid values.
    
    Args:
        values: List of values to clean.
        default: Default value for invalid entries.
    
    Returns:
        List of cleaned numeric values.
    """
    cleaned = []
    
    for value in values:
        try:
            if isinstance(value, str):
                cleaned.append(float(value.strip()))
            else:
                cleaned.append(float(value))
        except (ValueError, TypeError):
            cleaned.append(default)
    
    return cleaned

if __name__ == "__main__":
    # Example usage
    sample_data = [1, 2, 2, 3, 4, 4, 5, "1", "2.5", "invalid", 3.0]
    
    print("Original data:", sample_data)
    print("Unique values:", remove_duplicates(sample_data))
    
    numeric_data = clean_numeric_data(sample_data)
    print("Cleaned numeric data:", numeric_data)