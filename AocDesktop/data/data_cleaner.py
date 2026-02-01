
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def calculate_summary_statistics(data, column):
    """
    Calculate summary statistics for a specified column.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    dict: Dictionary containing mean, median, and standard deviation.
    """
    stats = {
        'mean': data[column].mean(),
        'median': data[column].median(),
        'std': data[column].std()
    }
    return stats

if __name__ == "__main__":
    import pandas as pd
    
    sample_data = pd.DataFrame({
        'values': np.random.normal(100, 15, 1000)
    })
    
    cleaned_data = remove_outliers_iqr(sample_data, 'values')
    stats = calculate_summary_statistics(cleaned_data, 'values')
    
    print(f"Original data shape: {sample_data.shape}")
    print(f"Cleaned data shape: {cleaned_data.shape}")
    print(f"Summary statistics: {stats}")import pandas as pd
import numpy as np

def clean_missing_data(df, strategy='mean', columns=None):
    """
    Clean missing data in a pandas DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    strategy (str): Strategy for handling missing values.
                    Options: 'mean', 'median', 'mode', 'drop', 'fill'.
    columns (list): List of column names to apply cleaning. If None, applies to all columns.

    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    if df.empty:
        return df

    if columns is None:
        columns = df.columns

    df_clean = df.copy()

    for col in columns:
        if col not in df_clean.columns:
            continue

        if df_clean[col].isnull().sum() == 0:
            continue

        if strategy == 'drop':
            df_clean = df_clean.dropna(subset=[col])
        elif strategy == 'mean':
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
        elif strategy == 'median':
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        elif strategy == 'mode':
            mode_val = df_clean[col].mode()
            if not mode_val.empty:
                df_clean[col] = df_clean[col].fillna(mode_val.iloc[0])
        elif strategy == 'fill':
            df_clean[col] = df_clean[col].fillna(method='ffill').fillna(method='bfill')
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.

    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.

    Returns:
    tuple: (bool, str) Validation result and message.
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

def main():
    # Example usage
    data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, 2, 3, np.nan, 5],
        'C': ['x', 'y', np.nan, 'z', 'w']
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame (mean strategy):")
    df_clean = clean_missing_data(df, strategy='mean')
    print(df_clean)
    
    is_valid, message = validate_dataframe(df_clean)
    print(f"\nValidation: {is_valid} - {message}")

if __name__ == "__main__":
    main()