
import re

def clean_string(text):
    """
    Cleans a string by:
    1. Removing leading and trailing whitespace.
    2. Converting multiple spaces/newlines/tabs to a single space.
    3. Converting the string to lowercase.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string")

    # Strip leading/trailing whitespace
    text = text.strip()

    # Replace any sequence of whitespace characters with a single space
    text = re.sub(r'\s+', ' ', text)

    # Convert to lowercase
    text = text.lower()

    return text

def normalize_list(string_list):
    """
    Applies clean_string to each element in a list.
    Returns a new list with cleaned strings.
    """
    if not isinstance(string_list, list):
        raise TypeError("Input must be a list")

    return [clean_string(item) for item in string_list]