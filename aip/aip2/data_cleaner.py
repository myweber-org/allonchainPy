
import re
import unicodedata

def clean_text(text):
    """
    Cleans and normalizes a given text string.
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove extra whitespace
    text = ' '.join(text.split())

    # Normalize unicode characters (e.g., convert accented characters to their base form)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')

    # Remove special characters, keep alphanumeric and basic punctuation
    text = re.sub(r'[^a-z0-9\s.,!?]', '', text)

    return text.strip()