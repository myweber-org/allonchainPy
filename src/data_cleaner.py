
import pandas as pd

def clean_dataset(df, drop_duplicates=True, normalize_cols=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing column names.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows.
        normalize_cols (bool): Whether to normalize column names to lowercase with underscores.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows.")
    
    if normalize_cols:
        original_cols = cleaned_df.columns.tolist()
        cleaned_df.columns = [
            col.lower().replace(' ', '_').replace('-', '_')
            for col in cleaned_df.columns
        ]
        print(f"Normalized column names: {dict(zip(original_cols, cleaned_df.columns))}")
    
    return cleaned_df

def validate_dataframe(df, required_cols=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_cols (list): List of required column names.
    
    Returns:
        dict: Validation results with keys 'is_valid' and 'messages'.
    """
    results = {
        'is_valid': True,
        'messages': []
    }
    
    if df.empty:
        results['is_valid'] = False
        results['messages'].append('DataFrame is empty.')
    
    if required_cols:
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            results['is_valid'] = False
            results['messages'].append(f'Missing required columns: {missing_cols}')
    
    if df.isnull().all().any():
        results['messages'].append('Warning: Some columns contain only null values.')
    
    return results

if __name__ == '__main__':
    sample_data = {
        'Product Name': ['A', 'B', 'A', 'C', 'B'],
        'Unit-Price': [10, 20, 10, 30, 20],
        'Quantity ': [1, 2, 1, 3, 2]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaning DataFrame...")
    
    cleaned = clean_dataset(df)
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    validation = validate_dataframe(cleaned, required_cols=['product_name', 'unit_price'])
    print(f"\nValidation: {validation}")