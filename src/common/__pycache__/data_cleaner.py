
import pandas as pd
import numpy as np

def clean_dataset(df, strategy='mean', outlier_threshold=3):
    """
    Clean a pandas DataFrame by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): Strategy for missing values: 'mean', 'median', 'mode', or 'drop'
    outlier_threshold (float): Number of standard deviations for outlier detection
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    categorical_cols = cleaned_df.select_dtypes(exclude=[np.number]).columns
    
    if strategy == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif strategy == 'mean':
        cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].mean())
    elif strategy == 'median':
        cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].median())
    elif strategy == 'mode':
        for col in cleaned_df.columns:
            if col in numeric_cols:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 0)
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 'Unknown')
    
    # Handle outliers for numeric columns
    for col in numeric_cols:
        if cleaned_df[col].dtype in [np.float64, np.int64]:
            mean = cleaned_df[col].mean()
            std = cleaned_df[col].std()
            
            if std > 0:  # Avoid division by zero
                z_scores = np.abs((cleaned_df[col] - mean) / std)
                outlier_mask = z_scores > outlier_threshold
                
                # Replace outliers with median
                if outlier_mask.any():
                    median_val = cleaned_df[col].median()
                    cleaned_df.loc[outlier_mask, col] = median_val
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataset(df, required_columns=None, min_rows=1):
    """
    Validate dataset structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Dataset is valid"

def get_dataset_stats(df):
    """
    Get basic statistics about the dataset.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    
    Returns:
    dict: Dictionary containing dataset statistics
    """
    stats = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': len(df.select_dtypes(exclude=[np.number]).columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    # Add column-wise missing values
    missing_by_col = df.isnull().sum()
    stats['missing_by_column'] = missing_by_col[missing_by_col > 0].to_dict()
    
    return stats

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'A': [1, 2, np.nan, 4, 100],  # Contains outlier and missing value
        'B': [5, 6, 7, np.nan, 9],
        'C': ['x', 'y', 'z', 'x', 'y']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\nDataset statistics:")
    print(get_dataset_stats(df))
    
    # Clean the dataset
    cleaned = clean_dataset(df, strategy='median', outlier_threshold=2)
    print("\nCleaned dataset:")
    print(cleaned)
    
    # Validate the cleaned dataset
    is_valid, message = validate_dataset(cleaned, required_columns=['A', 'B', 'C'])
    print(f"\nValidation: {message}")