
def remove_duplicates(input_list):
    """
    Remove duplicate elements from a list while preserving order.
    
    Args:
        input_list: A list containing elements (must be hashable).
    
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
    Clean data by removing duplicates and optionally filtering by count threshold.
    
    Args:
        data: List of items to clean.
        threshold: If provided, only items appearing at least threshold times are kept.
    
    Returns:
        Cleaned list according to specified rules.
    """
    if not data:
        return []
    
    if threshold is None:
        return remove_duplicates(data)
    
    from collections import Counter
    counter = Counter(data)
    filtered = [item for item in data if counter[item] >= threshold]
    return remove_duplicates(filtered)

if __name__ == "__main__":
    sample_data = [1, 2, 2, 3, 4, 4, 4, 5]
    print("Original:", sample_data)
    print("Deduplicated:", remove_duplicates(sample_data))
    print("Threshold >=2:", clean_data_with_threshold(sample_data, threshold=2))