
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

def clean_numeric_data(values, threshold=None):
    """
    Clean numeric data by removing None values and optionally
    filtering based on threshold.
    """
    cleaned = [v for v in values if v is not None]
    if threshold is not None:
        cleaned = [v for v in cleaned if v <= threshold]
    return cleaned

if __name__ == "__main__":
    # Example usage
    sample_data = [1, 2, 2, 3, 4, 4, 5]
    print("Original:", sample_data)
    print("Cleaned:", remove_duplicates(sample_data))
    
    numeric_data = [10, None, 20, 30, None, 40]
    print("Numeric cleaned:", clean_numeric_data(numeric_data, threshold=35))