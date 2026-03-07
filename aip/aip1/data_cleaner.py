
import pandas as pd
import numpy as np
from typing import List, Optional

def remove_duplicates(
    data: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = 'first',
    inplace: bool = False
) -> pd.DataFrame:
    """
    Remove duplicate rows from a pandas DataFrame.
    
    Parameters:
    data: Input DataFrame
    subset: Column labels to consider for identifying duplicates
    keep: Which duplicates to keep ('first', 'last', False)
    inplace: Whether to modify the DataFrame in place
    
    Returns:
    DataFrame with duplicates removed
    """
    if not inplace:
        data = data.copy()
    
    if subset is None:
        subset = data.columns.tolist()
    
    cleaned_data = data.drop_duplicates(subset=subset, keep=keep)
    
    if inplace:
        data.drop_duplicates(subset=subset, keep=keep, inplace=True)
        return data
    else:
        return cleaned_data

def validate_dataframe(data: pd.DataFrame) -> bool:
    """
    Basic validation of DataFrame structure.
    
    Parameters:
    data: DataFrame to validate
    
    Returns:
    Boolean indicating if DataFrame is valid
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if data.empty:
        print("Warning: DataFrame is empty")
        return False
    
    if data.isnull().all().all():
        print("Warning: All values in DataFrame are null")
        return False
    
    return True

def clean_numeric_columns(
    data: pd.DataFrame,
    columns: List[str],
    fill_value: float = 0.0
) -> pd.DataFrame:
    """
    Clean numeric columns by filling NaN values.
    
    Parameters:
    data: Input DataFrame
    columns: List of column names to clean
    fill_value: Value to use for filling NaN
    
    Returns:
    DataFrame with cleaned numeric columns
    """
    cleaned_data = data.copy()
    
    for col in columns:
        if col in cleaned_data.columns:
            cleaned_data[col] = pd.to_numeric(
                cleaned_data[col], 
                errors='coerce'
            ).fillna(fill_value)
    
    return cleaned_data

if __name__ == "__main__":
    # Example usage
    sample_data = pd.DataFrame({
        'id': [1, 2, 2, 3, 4],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', 'David'],
        'score': [85, 90, 90, 78, np.nan],
        'age': [25, 30, 30, 35, 40]
    })
    
    print("Original data:")
    print(sample_data)
    print("\nAfter removing duplicates:")
    cleaned = remove_duplicates(sample_data, subset=['id', 'name'])
    print(cleaned)
    
    print("\nCleaning numeric columns:")
    numeric_cleaned = clean_numeric_columns(cleaned, columns=['score'])
    print(numeric_cleaned)