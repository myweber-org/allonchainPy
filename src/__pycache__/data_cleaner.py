
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

def clean_numeric_strings(string_list):
    """
    Clean a list of strings by removing non-numeric characters
    and converting to integers where possible.
    """
    cleaned = []
    for s in string_list:
        try:
            numeric_part = ''.join(filter(str.isdigit, s))
            if numeric_part:
                cleaned.append(int(numeric_part))
        except ValueError:
            continue
    return cleaned

def filter_by_threshold(values, threshold, above=True):
    """
    Filter values based on a threshold.
    If above is True, keep values >= threshold.
    If above is False, keep values < threshold.
    """
    if above:
        return [v for v in values if v >= threshold]
    else:
        return [v for v in values if v < threshold]