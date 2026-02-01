
def remove_duplicates(input_list):
    """
    Remove duplicate elements from a list while preserving order.
    
    Args:
        input_list (list): The list from which duplicates are to be removed.
    
    Returns:
        list: A new list with duplicates removed.
    """
    seen = set()
    result = []
    
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    
    return result

def clean_data_with_threshold(data, threshold=None):
    """
    Clean data by removing duplicates, optionally with a frequency threshold.
    
    Args:
        data (list): Input data list.
        threshold (int, optional): Minimum frequency to keep an item. Defaults to None.
    
    Returns:
        list: Cleaned data list.
    """
    cleaned = remove_duplicates(data)
    
    if threshold is not None and threshold > 0:
        from collections import Counter
        counts = Counter(data)
        cleaned = [item for item in cleaned if counts[item] >= threshold]
    
    return cleaned

if __name__ == "__main__":
    sample_data = [1, 2, 2, 3, 4, 4, 4, 5, 1, 6]
    print("Original data:", sample_data)
    print("Without duplicates:", remove_duplicates(sample_data))
    print("With threshold 2:", clean_data_with_threshold(sample_data, threshold=2))