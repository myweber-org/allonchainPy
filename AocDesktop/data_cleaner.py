import re

def clean_string(text):
    """
    Cleans a string by:
    1. Converting to lowercase.
    2. Removing leading/trailing whitespace.
    3. Replacing multiple spaces with a single space.
    4. Removing non-alphanumeric characters (except basic punctuation).
    """
    if not isinstance(text, str):
        return text

    # Convert to lowercase and strip whitespace
    text = text.lower().strip()

    # Replace multiple spaces/newlines/tabs with a single space
    text = re.sub(r'\s+', ' ', text)

    # Remove characters that are not alphanumeric, space, or basic punctuation
    # Keeps: letters, numbers, spaces, . , ! ? ' " - ( )
    text = re.sub(r'[^a-z0-9\s.,!?\'\"\-()]', '', text)

    return text

def normalize_whitespace(text):
    """A simpler function that only normalizes excessive whitespace."""
    if not isinstance(text, str):
        return text
    return ' '.join(text.split())