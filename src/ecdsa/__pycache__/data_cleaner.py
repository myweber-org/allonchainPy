
import pandas as pd
import numpy as np

def clean_dataset(df, missing_strategy='mean', outlier_method='iqr', columns=None):
    """
    Clean a pandas DataFrame by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    missing_strategy (str): Strategy for missing values - 'mean', 'median', 'mode', or 'drop'
    outlier_method (str): Method for outlier detection - 'iqr' or 'zscore'
    columns (list): Specific columns to clean, if None clean all numeric columns
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if columns is None:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    else:
        numeric_cols = [col for col in columns if col in df_clean.columns]
    
    # Handle missing values
    for col in numeric_cols:
        if df_clean[col].isnull().any():
            if missing_strategy == 'mean':
                fill_value = df_clean[col].mean()
            elif missing_strategy == 'median':
                fill_value = df_clean[col].median()
            elif missing_strategy == 'mode':
                fill_value = df_clean[col].mode()[0]
            elif missing_strategy == 'drop':
                df_clean = df_clean.dropna(subset=[col])
                continue
            else:
                raise ValueError(f"Unknown missing strategy: {missing_strategy}")
            
            df_clean[col] = df_clean[col].fillna(fill_value)
    
    # Handle outliers
    for col in numeric_cols:
        if outlier_method == 'iqr':
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers
            df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
            
        elif outlier_method == 'zscore':
            mean_val = df_clean[col].mean()
            std_val = df_clean[col].std()
            
            # Calculate z-scores
            z_scores = np.abs((df_clean[col] - mean_val) / std_val)
            
            # Cap values beyond 3 standard deviations
            mask = z_scores > 3
            df_clean.loc[mask, col] = np.where(
                df_clean.loc[mask, col] > mean_val,
                mean_val + 3 * std_val,
                mean_val - 3 * std_val
            )
    
    return df_clean

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
    if len(df) < min_rows:
        return False, f"Dataset must have at least {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Dataset is valid"

# Example usage function
def demonstrate_cleaning():
    """Demonstrate the data cleaning functionality."""
    # Create sample data with missing values and outliers
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.normal(50, 10, 100),
        'feature_c': np.random.normal(0, 1, 100)
    }
    
    # Add some missing values
    for col in data:
        mask = np.random.random(100) < 0.1
        data[col][mask] = np.nan
    
    # Add some outliers
    data['feature_a'][0] = 500  # Extreme outlier
    data['feature_b'][1] = -100  # Extreme negative outlier
    
    df = pd.DataFrame(data)
    
    print("Original dataset shape:", df.shape)
    print("Missing values per column:")
    print(df.isnull().sum())
    
    # Clean the dataset
    df_clean = clean_dataset(df, missing_strategy='median', outlier_method='iqr')
    
    print("\nCleaned dataset shape:", df_clean.shape)
    print("Missing values after cleaning:")
    print(df_clean.isnull().sum())
    
    # Validate the cleaned dataset
    is_valid, message = validate_dataset(df_clean, min_rows=50)
    print(f"\nValidation: {message}")
    
    return df_clean

if __name__ == "__main__":
    cleaned_data = demonstrate_cleaning()
    print("\nCleaning demonstration completed successfully.")