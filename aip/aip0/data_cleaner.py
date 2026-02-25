import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Parameters:
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

def calculate_summary_statistics(df):
    """
    Calculate summary statistics for numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    
    Returns:
    pd.DataFrame: Summary statistics
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    summary = df[numeric_cols].describe()
    
    return summary

def clean_missing_values(df, strategy='mean'):
    """
    Handle missing values in numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): Strategy for imputation ('mean', 'median', 'drop')
    
    Returns:
    pd.DataFrame: DataFrame with handled missing values
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if strategy == 'drop':
        cleaned_df = df.dropna(subset=numeric_cols)
    elif strategy == 'mean':
        cleaned_df = df.copy()
        for col in numeric_cols:
            cleaned_df[col].fillna(df[col].mean(), inplace=True)
    elif strategy == 'median':
        cleaned_df = df.copy()
        for col in numeric_cols:
            cleaned_df[col].fillna(df[col].median(), inplace=True)
    else:
        raise ValueError("Strategy must be 'mean', 'median', or 'drop'")
    
    return cleaned_df

def normalize_data(df, columns=None):
    """
    Normalize specified columns using min-max scaling.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of columns to normalize
    
    Returns:
    pd.DataFrame: DataFrame with normalized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    normalized_df = df.copy()
    
    for col in columns:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            
            if max_val != min_val:
                normalized_df[col] = (df[col] - min_val) / (max_val - min_val)
            else:
                normalized_df[col] = 0
    
    return normalized_df

def process_data_pipeline(df, outlier_columns=None, missing_strategy='mean', normalize_cols=None):
    """
    Complete data processing pipeline.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    outlier_columns (list): Columns to remove outliers from
    missing_strategy (str): Strategy for handling missing values
    normalize_cols (list): Columns to normalize
    
    Returns:
    pd.DataFrame: Processed DataFrame
    """
    processed_df = df.copy()
    
    if outlier_columns:
        for col in outlier_columns:
            if col in processed_df.columns:
                processed_df = remove_outliers_iqr(processed_df, col)
    
    processed_df = clean_missing_values(processed_df, strategy=missing_strategy)
    
    if normalize_cols:
        processed_df = normalize_data(processed_df, columns=normalize_cols)
    
    return processed_df

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 3, 4, 5, 100],
        'B': [10, 20, 30, 40, 50, 60],
        'C': [1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nSummary Statistics:")
    print(calculate_summary_statistics(df))
    
    processed = process_data_pipeline(
        df, 
        outlier_columns=['A'],
        missing_strategy='mean',
        normalize_cols=['B', 'C']
    )
    
    print("\nProcessed DataFrame:")
    print(processed)
import pandas as pd
import numpy as np

def clean_dataframe(df):
    """
    Clean a pandas DataFrame by removing duplicates and standardizing column names.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()
    
    # Standardize column names: lowercase and replace spaces with underscores
    df_cleaned.columns = df_cleaned.columns.str.lower().str.replace(' ', '_')
    
    # Remove rows with missing values in critical columns
    critical_columns = ['id', 'name', 'value']
    existing_critical = [col for col in critical_columns if col in df_cleaned.columns]
    if existing_critical:
        df_cleaned = df_cleaned.dropna(subset=existing_critical)
    
    return df_cleaned

def validate_data(df, required_columns):
    """
    Validate that the DataFrame contains required columns and has no negative values.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for negative values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if (df[col] < 0).any():
            print(f"Warning: Negative values found in column '{col}'")
    
    return True

def process_data(input_file, output_file):
    """
    Main function to process data from input file and save cleaned data to output file.
    """
    try:
        # Read input data
        df = pd.read_csv(input_file)
        
        # Clean the data
        df_cleaned = clean_dataframe(df)
        
        # Validate the cleaned data
        required_cols = ['id', 'value']
        validate_data(df_cleaned, required_cols)
        
        # Save cleaned data
        df_cleaned.to_csv(output_file, index=False)
        print(f"Data cleaned successfully. Saved to {output_file}")
        
        return df_cleaned
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except Exception as e:
        print(f"Error processing data: {str(e)}")

if __name__ == "__main__":
    # Example usage
    input_path = "raw_data.csv"
    output_path = "cleaned_data.csv"
    
    cleaned_df = process_data(input_path, output_path)
    if cleaned_df is not None:
        print(f"Cleaned data shape: {cleaned_df.shape}")
        print("Sample of cleaned data:")
        print(cleaned_df.head())
import pandas as pd

def clean_dataframe(df, drop_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
        fill_missing (str or dict): Method to fill missing values. 
            Options: 'mean', 'median', 'mode', or a dictionary of column:value pairs.
            If None, missing values are not filled.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing is not None:
        if isinstance(fill_missing, dict):
            cleaned_df = cleaned_df.fillna(fill_missing)
        elif fill_missing == 'mean':
            cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
        elif fill_missing == 'median':
            cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
        elif fill_missing == 'mode':
            for col in cleaned_df.columns:
                if cleaned_df[col].dtype == 'object':
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame for required columns and basic integrity.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"