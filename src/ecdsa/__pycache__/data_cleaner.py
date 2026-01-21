
import re

def clean_text(text):
    """
    Cleans the input text by:
    1. Removing leading/trailing whitespace.
    2. Converting multiple spaces/newlines/tabs to a single space.
    3. Converting the text to lowercase.
    
    Args:
        text (str): The input text to be cleaned.
    
    Returns:
        str: The cleaned text.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string")
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Replace multiple whitespace characters (spaces, newlines, tabs) with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Convert to lowercase
    text = text.lower()
    
    return text