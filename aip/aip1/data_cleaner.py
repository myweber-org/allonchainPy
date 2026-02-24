
def remove_duplicates(data_list):
    """
    Remove duplicate entries from a list while preserving order.
    
    Args:
        data_list: A list containing elements that may have duplicates.
    
    Returns:
        A new list with duplicates removed.
    """
    seen = set()
    result = []
    for item in data_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def clean_data_with_key(data_list, key_func=None):
    """
    Remove duplicates based on a key function.
    
    Args:
        data_list: A list containing elements.
        key_func: A function to extract comparison key from each element.
    
    Returns:
        A new list with duplicates removed based on the key.
    """
    if key_func is None:
        return remove_duplicates(data_list)
    
    seen = set()
    result = []
    for item in data_list:
        key = key_func(item)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result

if __name__ == "__main__":
    sample_data = [1, 2, 2, 3, 4, 4, 5]
    cleaned = remove_duplicates(sample_data)
    print(f"Original: {sample_data}")
    print(f"Cleaned: {cleaned}")
    
    sample_objects = [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"},
        {"id": 1, "name": "Alice"},
        {"id": 3, "name": "Charlie"}
    ]
    cleaned_objects = clean_data_with_key(sample_objects, key_func=lambda x: x["id"])
    print(f"Original objects: {sample_objects}")
    print(f"Cleaned objects: {cleaned_objects}")