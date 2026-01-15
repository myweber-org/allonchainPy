def remove_duplicates(data_list):
    seen = set()
    result = []
    for item in data_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def clean_data_with_order(data_list, key=None):
    if key is None:
        key = lambda x: x
    seen = set()
    result = []
    for item in data_list:
        identifier = key(item)
        if identifier not in seen:
            seen.add(identifier)
            result.append(item)
    return result

if __name__ == "__main__":
    sample_data = [1, 2, 2, 3, 4, 4, 5]
    cleaned = remove_duplicates(sample_data)
    print(f"Original: {sample_data}")
    print(f"Cleaned: {cleaned}")