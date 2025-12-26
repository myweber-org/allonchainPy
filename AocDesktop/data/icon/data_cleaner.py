
import re
import json
from typing import Dict, Any, Optional, List

def sanitize_string(input_string: str) -> str:
    """Remove extra whitespace and normalize line endings."""
    if not isinstance(input_string, str):
        raise TypeError("Input must be a string")
    cleaned = re.sub(r'\s+', ' ', input_string.strip())
    return cleaned.replace('\r\n', '\n').replace('\r', '\n')

def validate_email(email: str) -> bool:
    """Check if the provided string is a valid email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def parse_json_safe(json_string: str) -> Optional[Dict[str, Any]]:
    """Safely parse a JSON string, returning None on failure."""
    try:
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError):
        return None

def filter_list_by_prefix(items: List[str], prefix: str) -> List[str]:
    """Filter a list of strings, keeping only those starting with a prefix."""
    return [item for item in items if item.startswith(prefix)]

def calculate_checksum(data: str) -> str:
    """Calculate a simple checksum for a string."""
    if not data:
        return '0'
    hash_val = 0
    for char in data:
        hash_val = (hash_val * 31 + ord(char)) & 0xFFFFFFFF
    return format(hash_val, '08x')