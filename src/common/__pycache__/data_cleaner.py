import pandas as pd
import numpy as np

def clean_csv_data(input_file, output_file, strategy='mean'):
    """
    Clean a CSV file by handling missing values.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to save cleaned CSV file
        strategy (str): Strategy for handling missing values.
                       Options: 'mean', 'median', 'mode', 'drop'
    """
    try:
        df = pd.read_csv(input_file)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].isnull().any():
                if strategy == 'mean':
                    fill_value = df[col].mean()
                elif strategy == 'median':
                    fill_value = df[col].median()
                elif strategy == 'mode':
                    fill_value = df[col].mode()[0]
                elif strategy == 'drop':
                    df = df.dropna(subset=[col])
                    continue
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")
                
                df[col] = df[col].fillna(fill_value)
        
        df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to: {output_file}")
        print(f"Original shape: {df.shape}")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except pd.errors.EmptyDataError:
        print("Error: Input file is empty.")
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")

def validate_dataframe(df):
    """
    Validate dataframe for common data quality issues.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
    
    Returns:
        dict: Dictionary containing validation results
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(df.select_dtypes(include=['object']).columns)
    }
    
    return validation_results

if __name__ == "__main__":
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    
    clean_csv_data(input_csv, output_csv, strategy='median')
    
    df_cleaned = pd.read_csv(output_csv)
    validation = validate_dataframe(df_cleaned)
    
    print("\nData Validation Results:")
    for key, value in validation.items():
        print(f"{key}: {value}")
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
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
    
    return filtered_df.reset_index(drop=True)

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': len(df[column])
    }
    
    return stats

def process_dataset(file_path, column_to_clean):
    """
    Load a dataset, clean outliers, and return cleaned data with statistics.
    
    Parameters:
    file_path (str): Path to CSV file
    column_to_clean (str): Column name to process
    
    Returns:
    tuple: (cleaned DataFrame, original stats, cleaned stats)
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    
    original_stats = calculate_summary_statistics(df, column_to_clean)
    cleaned_df = remove_outliers_iqr(df, column_to_clean)
    cleaned_stats = calculate_summary_statistics(cleaned_df, column_to_clean)
    
    return cleaned_df, original_stats, cleaned_stats