import re

def clean_string(text):
    """
    Cleans a string by:
    1. Stripping leading/trailing whitespace.
    2. Replacing multiple spaces/newlines/tabs with a single space.
    3. Converting to lowercase.
    Returns an empty string if input is not a string.
    """
    if not isinstance(text, str):
        return ''
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    # Replace any sequence of whitespace characters with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Convert to lowercase
    text = text.lower()
    
    return text

def normalize_data(data):
    """
    Accepts a list of strings or a single string.
    Returns a list of cleaned strings.
    """
    if isinstance(data, str):
        data = [data]
    
    cleaned_list = []
    for item in data:
        cleaned_item = clean_string(item)
        cleaned_list.append(cleaned_item)
    
    return cleaned_list