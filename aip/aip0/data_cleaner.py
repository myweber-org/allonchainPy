
def remove_duplicates(input_list):
    """
    Remove duplicate elements from a list while preserving order.
    
    Args:
        input_list: A list that may contain duplicate elements.
    
    Returns:
        A new list with duplicates removed.
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
    Clean data by removing duplicates, optionally filtering by frequency threshold.
    
    Args:
        data: List of data items.
        threshold: Minimum frequency count to keep an item (inclusive).
    
    Returns:
        Cleaned list of items.
    """
    from collections import Counter
    
    if not data:
        return []
    
    counter = Counter(data)
    
    if threshold is None:
        return remove_duplicates(data)
    
    cleaned = [item for item in data if counter[item] >= threshold]
    return remove_duplicates(cleaned)

if __name__ == "__main__":
    sample_data = [1, 2, 2, 3, 4, 4, 4, 5, 1]
    print(f"Original data: {sample_data}")
    print(f"Without duplicates: {remove_duplicates(sample_data)}")
    print(f"Threshold >= 2: {clean_data_with_threshold(sample_data, threshold=2)}")