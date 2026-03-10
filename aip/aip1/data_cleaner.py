
import pandas as pd
import numpy as np
from typing import Optional

def clean_csv_data(
    input_path: str,
    output_path: Optional[str] = None,
    missing_strategy: str = 'mean',
    drop_threshold: float = 0.5
) -> pd.DataFrame:
    """
    Clean CSV data by handling missing values and removing high-missing columns.
    
    Parameters:
    -----------
    input_path : str
        Path to input CSV file
    output_path : Optional[str]
        Path to save cleaned data (if None, returns DataFrame only)
    missing_strategy : str
        Strategy for filling missing values ('mean', 'median', 'mode', 'drop')
    drop_threshold : float
        Threshold for dropping columns with missing values (0.0 to 1.0)
    
    Returns:
    --------
    pd.DataFrame
        Cleaned DataFrame
    """
    
    # Read input data
    df = pd.read_csv(input_path)
    
    # Calculate missing percentage per column
    missing_percent = df.isnull().sum() / len(df)
    
    # Drop columns with missing values above threshold
    columns_to_drop = missing_percent[missing_percent > drop_threshold].index
    df = df.drop(columns=columns_to_drop)
    
    # Handle remaining missing values
    if missing_strategy == 'mean':
        df = df.fillna(df.select_dtypes(include=[np.number]).mean())
    elif missing_strategy == 'median':
        df = df.fillna(df.select_dtypes(include=[np.number]).median())
    elif missing_strategy == 'mode':
        df = df.fillna(df.mode().iloc[0])
    elif missing_strategy == 'drop':
        df = df.dropna()
    else:
        raise ValueError(f"Unknown missing strategy: {missing_strategy}")
    
    # Reset index after cleaning
    df = df.reset_index(drop=True)
    
    # Save to output path if provided
    if output_path:
        df.to_csv(output_path, index=False)
    
    return df

def validate_dataframe(df: pd.DataFrame) -> dict:
    """
    Validate DataFrame structure and content.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to validate
    
    Returns:
    --------
    dict
        Validation results
    """
    validation_results = {
        'row_count': len(df),
        'column_count': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict(),
        'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(df.select_dtypes(include=['object']).columns)
    }
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, 2, 3, np.nan, 5],
        'C': [1, 1, 1, 1, 1],
        'D': ['a', 'b', 'c', 'd', 'e']
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_csv_data(
        'sample_data.csv',
        missing_strategy='mean',
        drop_threshold=0.3
    )
    
    validation = validate_dataframe(cleaned_df)
    print(f"Cleaned data shape: {cleaned_df.shape}")
    print(f"Validation results: {validation}")