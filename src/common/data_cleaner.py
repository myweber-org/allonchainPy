
def remove_duplicates_preserve_order(input_list):
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return resultdef remove_duplicates(input_list):
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
        numeric_part = ''.join(filter(str.isdigit, s))
        if numeric_part:
            cleaned.append(int(numeric_part))
    return cleaned

def validate_email_format(email_list):
    """
    Validate email formats in a list and return only valid emails.
    Basic validation checking for '@' and '.' in the domain.
    """
    valid_emails = []
    for email in email_list:
        if '@' in email and '.' in email.split('@')[-1]:
            valid_emails.append(email.strip().lower())
    return valid_emails

if __name__ == "__main__":
    # Example usage
    sample_data = [1, 2, 2, 3, 4, 4, 5]
    print("Original:", sample_data)
    print("Cleaned:", remove_duplicates(sample_data))
    
    sample_strings = ["abc123", "456def", "ghi789", "jkl"]
    print("Numeric strings cleaned:", clean_numeric_strings(sample_strings))
    
    emails = ["test@example.com", "invalid-email", "user@domain.org"]
    print("Valid emails:", validate_email_format(emails))