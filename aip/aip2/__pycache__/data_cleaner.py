
def remove_duplicates(data_list):
    """
    Remove duplicate entries from a list while preserving order.
    
    Args:
        data_list: List of elements (must be hashable)
    
    Returns:
        List with duplicates removed
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
        values: List of numeric values or strings
        default: Default value for invalid entries
    
    Returns:
        List of cleaned numeric values
    """
    cleaned = []
    
    for val in values:
        try:
            cleaned.append(float(val))
        except (ValueError, TypeError):
            cleaned.append(default)
    
    return cleaned

if __name__ == "__main__":
    # Test the functions
    test_data = [1, 2, 2, 3, 4, 4, 5]
    print("Original:", test_data)
    print("Cleaned:", remove_duplicates(test_data))
    
    test_numeric = ["1.5", "2.3", "invalid", "4.7", None]
    print("Numeric original:", test_numeric)
    print("Numeric cleaned:", clean_numeric_data(test_numeric))