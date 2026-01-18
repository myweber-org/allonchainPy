import pandas as pd

def clean_dataset(df, remove_duplicates=True, fill_method=None):
    """
    Clean a pandas DataFrame by handling missing values and removing duplicates.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        remove_duplicates (bool): Whether to remove duplicate rows
        fill_method (str or None): Method to fill missing values - 'mean', 'median', 'mode', or None
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if fill_method:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        
        if fill_method == 'mean':
            for col in numeric_cols:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
        elif fill_method == 'median':
            for col in numeric_cols:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        elif fill_method == 'mode':
            for col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
    else:
        # Drop rows with all NaN values
        cleaned_df = cleaned_df.dropna(how='all')
    
    # Remove duplicates
    if remove_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate a DataFrame for basic integrity checks.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        dict: Dictionary with validation results
    """
    validation_results = {
        'is_valid': True,
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'column_types': df.dtypes.to_dict()
    }
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        validation_results['missing_required_columns'] = missing_columns
        validation_results['is_valid'] = len(missing_columns) == 0
    
    return validation_results

def sample_data(df, sample_size=0.1, random_state=42):
    """
    Create a random sample from the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        sample_size (float or int): Fraction or number of samples
        random_state (int): Random seed for reproducibility
    
    Returns:
        pd.DataFrame: Sampled DataFrame
    """
    if isinstance(sample_size, float) and 0 < sample_size < 1:
        return df.sample(frac=sample_size, random_state=random_state)
    elif isinstance(sample_size, int) and sample_size > 0:
        return df.sample(n=min(sample_size, len(df)), random_state=random_state)
    else:
        raise ValueError("sample_size must be a float between 0 and 1 or a positive integer")