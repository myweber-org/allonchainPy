import pandas as pd
import numpy as np

def clean_dataframe(df, missing_strategy='mean', outlier_threshold=3):
    """
    Clean a pandas DataFrame by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    missing_strategy (str): Strategy for handling missing values. 
                            Options: 'mean', 'median', 'drop', 'fill_zero'.
    outlier_threshold (float): Number of standard deviations for outlier detection.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    
    if missing_strategy == 'mean':
        cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(
            cleaned_df[numeric_cols].mean()
        )
    elif missing_strategy == 'median':
        cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(
            cleaned_df[numeric_cols].median()
        )
    elif missing_strategy == 'fill_zero':
        cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(0)
    elif missing_strategy == 'drop':
        cleaned_df = cleaned_df.dropna(subset=numeric_cols)
    
    # Handle outliers using Z-score method
    if outlier_threshold > 0:
        for col in numeric_cols:
            z_scores = np.abs((cleaned_df[col] - cleaned_df[col].mean()) / 
                             cleaned_df[col].std())
            outlier_mask = z_scores > outlier_threshold
            
            if outlier_mask.any():
                # Cap outliers at threshold * standard deviation
                upper_bound = cleaned_df[col].mean() + outlier_threshold * cleaned_df[col].std()
                lower_bound = cleaned_df[col].mean() - outlier_threshold * cleaned_df[col].std()
                cleaned_df.loc[outlier_mask, col] = np.where(
                    cleaned_df.loc[outlier_mask, col] > upper_bound,
                    upper_bound,
                    lower_bound
                )
    
    # Clean non-numeric columns by filling with mode
    non_numeric_cols = cleaned_df.select_dtypes(exclude=[np.number]).columns
    for col in non_numeric_cols:
        if cleaned_df[col].isna().any():
            mode_value = cleaned_df[col].mode()
            if not mode_value.empty:
                cleaned_df[col] = cleaned_df[col].fillna(mode_value.iloc[0])
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    min_rows (int): Minimum number of rows required.
    
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
    
    return True, "DataFrame is valid"

# Example usage function
def process_example_data():
    """Example function demonstrating data cleaning."""
    np.random.seed(42)
    
    # Create sample data with missing values and outliers
    data = {
        'temperature': np.random.normal(25, 5, 100),
        'humidity': np.random.normal(60, 10, 100),
        'location': ['A', 'B', 'C'] * 33 + ['A'],
        'pressure': np.random.normal(1013, 50, 100)
    }
    
    df = pd.DataFrame(data)
    
    # Introduce missing values
    df.loc[10:15, 'temperature'] = np.nan
    df.loc[20:25, 'humidity'] = np.nan
    df.loc[5, 'location'] = np.nan
    
    # Introduce outliers
    df.loc[0, 'temperature'] = 100
    df.loc[1, 'pressure'] = 2000
    
    print("Original DataFrame shape:", df.shape)
    print("Missing values:\n", df.isna().sum())
    
    # Clean the data
    cleaned_df = clean_dataframe(df, missing_strategy='median', outlier_threshold=3)
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("Missing values after cleaning:\n", cleaned_df.isna().sum())
    
    # Validate the cleaned data
    is_valid, message = validate_dataframe(cleaned_df, min_rows=50)
    print(f"\nValidation: {message}")
    
    return cleaned_df

if __name__ == "__main__":
    result_df = process_example_data()
    print("\nFirst 5 rows of cleaned data:")
    print(result_df.head())