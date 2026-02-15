
import pandas as pd
import numpy as np

def clean_dataset(df, missing_strategy='mean', outlier_threshold=3):
    """
    Clean a dataset by handling missing values and removing outliers.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        missing_strategy (str): Strategy for handling missing values 
                               ('mean', 'median', 'mode', 'drop')
        outlier_threshold (float): Z-score threshold for outlier detection
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    
    if missing_strategy == 'mean':
        for col in numeric_cols:
            cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
    elif missing_strategy == 'median':
        for col in numeric_cols:
            cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
    elif missing_strategy == 'mode':
        for col in numeric_cols:
            cleaned_df[col].fillna(cleaned_df[col].mode()[0], inplace=True)
    elif missing_strategy == 'drop':
        cleaned_df.dropna(inplace=True)
    
    # Remove outliers using Z-score method
    if outlier_threshold > 0:
        z_scores = np.abs((cleaned_df[numeric_cols] - cleaned_df[numeric_cols].mean()) / 
                          cleaned_df[numeric_cols].std())
        outlier_mask = (z_scores < outlier_threshold).all(axis=1)
        cleaned_df = cleaned_df[outlier_mask].reset_index(drop=True)
    
    # Clean column names
    cleaned_df.columns = [col.strip().lower().replace(' ', '_') for col in cleaned_df.columns]
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

# Example usage function
def process_example_data():
    """Example function demonstrating data cleaning."""
    # Create sample data with missing values and outliers
    np.random.seed(42)
    data = {
        'age': np.random.normal(30, 10, 100),
        'salary': np.random.normal(50000, 15000, 100),
        'experience': np.random.randint(0, 20, 100)
    }
    
    # Introduce missing values
    for col in data:
        mask = np.random.random(100) < 0.1
        data[col][mask] = np.nan
    
    # Introduce outliers
    data['salary'][0] = 1000000  # Extreme outlier
    
    df = pd.DataFrame(data)
    
    print("Original DataFrame shape:", df.shape)
    print("Missing values per column:")
    print(df.isnull().sum())
    
    # Clean the data
    cleaned_df = clean_dataset(df, missing_strategy='median', outlier_threshold=3)
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("Missing values after cleaning:")
    print(cleaned_df.isnull().sum())
    
    # Validate the cleaned data
    is_valid, message = validate_dataframe(cleaned_df)
    print(f"\nData validation: {is_valid} - {message}")
    
    return cleaned_df

if __name__ == "__main__":
    result = process_example_data()
    print("\nData cleaning completed successfully.")