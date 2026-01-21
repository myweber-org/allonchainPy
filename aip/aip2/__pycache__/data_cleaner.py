
import re

def clean_text(text):
    """
    Clean and normalize a given text string.
    
    Args:
        text (str): The input text to be cleaned.
    
    Returns:
        str: The cleaned text with extra whitespace removed and converted to lowercase.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string")
    
    # Remove leading and trailing whitespace
    text = text.strip()
    
    # Replace multiple spaces or newlines with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Convert to lowercase
    text = text.lower()
    
    return text