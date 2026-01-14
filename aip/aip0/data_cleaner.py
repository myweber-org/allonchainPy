
def remove_duplicates(data_list):
    """
    Remove duplicate entries from a list while preserving order.
    
    Args:
        data_list (list): Input list potentially containing duplicates.
    
    Returns:
        list: List with duplicates removed.
    """
    seen = set()
    result = []
    
    for item in data_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    
    return result

def clean_numeric_strings(data_list):
    """
    Clean list by converting numeric strings to integers.
    
    Args:
        data_list (list): List containing mixed string and numeric values.
    
    Returns:
        list: List with numeric strings converted to integers.
    """
    cleaned = []
    
    for item in data_list:
        if isinstance(item, str) and item.isdigit():
            cleaned.append(int(item))
        else:
            cleaned.append(item)
    
    return cleaned

def filter_by_type(data_list, data_type):
    """
    Filter list to include only items of specified type.
    
    Args:
        data_list (list): Input list with mixed types.
        data_type (type): Type to filter by.
    
    Returns:
        list: Filtered list containing only items of specified type.
    """
    return [item for item in data_list if isinstance(item, data_type)]

if __name__ == "__main__":
    # Example usage
    sample_data = [1, 2, 2, 3, "4", "4", 5, "hello", 5]
    
    print("Original:", sample_data)
    print("Without duplicates:", remove_duplicates(sample_data))
    print("Cleaned numeric strings:", clean_numeric_strings(sample_data))
    print("Integers only:", filter_by_type(sample_data, int))