
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to process
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def clean_dataset(df, numeric_columns=None):
    """
    Clean dataset by removing outliers from specified numeric columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        numeric_columns (list): List of numeric column names to clean
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in cleaned_df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'id': range(1, 101),
        'value': np.random.randn(100) * 10 + 50
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[95:99, 'value'] = [200, -100, 300, 150, 250]
    
    print(f"Original dataset shape: {df.shape}")
    cleaned_df = clean_dataset(df, ['value'])
    print(f"Cleaned dataset shape: {cleaned_df.shape}")import pandas as pd
import numpy as np

def clean_csv_data(input_file, output_file, missing_strategy='mean'):
    """
    Load a CSV file, clean missing values, and save cleaned data.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to save cleaned CSV file
        missing_strategy (str): Strategy for handling missing values
                               'mean', 'median', 'drop', or 'zero'
    """
    try:
        df = pd.read_csv(input_file)
        print(f"Loaded data with shape: {df.shape}")
        
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"Found {missing_count} missing values")
            
            if missing_strategy == 'mean':
                df = df.fillna(df.mean(numeric_only=True))
            elif missing_strategy == 'median':
                df = df.fillna(df.median(numeric_only=True))
            elif missing_strategy == 'drop':
                df = df.dropna()
            elif missing_strategy == 'zero':
                df = df.fillna(0)
            else:
                raise ValueError(f"Unknown strategy: {missing_strategy}")
            
            print(f"Missing values handled using '{missing_strategy}' strategy")
        
        df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to: {output_file}")
        print(f"Final shape: {df.shape}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_data(df, required_columns=None):
    """
    Validate data quality after cleaning.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        dict: Validation results
    """
    validation_results = {
        'has_data': not df.empty,
        'row_count': len(df),
        'column_count': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        validation_results['missing_columns'] = missing_cols
        validation_results['all_columns_present'] = len(missing_cols) == 0
    
    return validation_results

if __name__ == "__main__":
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    
    cleaned_df = clean_csv_data(input_csv, output_csv, missing_strategy='median')
    
    if cleaned_df is not None:
        validation = validate_data(cleaned_df)
        print("\nData Validation Results:")
        for key, value in validation.items():
            print(f"{key}: {value}")