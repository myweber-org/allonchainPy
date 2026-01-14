
import pandas as pd
import numpy as np

def clean_dataset(df, columns_to_check=None, fill_missing='mean', remove_duplicates=True):
    """
    Clean a pandas DataFrame by handling missing values and removing duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    columns_to_check (list): Specific columns to check for missing values, if None checks all columns
    fill_missing (str): Method to fill missing values - 'mean', 'median', 'mode', or 'drop'
    remove_duplicates (bool): Whether to remove duplicate rows
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    
    df_clean = df.copy()
    
    # Remove duplicates if requested
    if remove_duplicates:
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed = initial_rows - len(df_clean)
        print(f"Removed {removed} duplicate rows")
    
    # Handle missing values
    if columns_to_check is None:
        columns_to_check = df_clean.columns
    
    missing_counts = df_clean[columns_to_check].isnull().sum()
    total_missing = missing_counts.sum()
    
    if total_missing > 0:
        print(f"Found {total_missing} missing values")
        
        for col in columns_to_check:
            missing_count = missing_counts[col]
            if missing_count > 0:
                if fill_missing == 'mean' and pd.api.types.is_numeric_dtype(df_clean[col]):
                    fill_value = df_clean[col].mean()
                    df_clean[col] = df_clean[col].fillna(fill_value)
                    print(f"Filled {missing_count} missing values in '{col}' with mean: {fill_value:.2f}")
                
                elif fill_missing == 'median' and pd.api.types.is_numeric_dtype(df_clean[col]):
                    fill_value = df_clean[col].median()
                    df_clean[col] = df_clean[col].fillna(fill_value)
                    print(f"Filled {missing_count} missing values in '{col}' with median: {fill_value:.2f}")
                
                elif fill_missing == 'mode':
                    fill_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else np.nan
                    df_clean[col] = df_clean[col].fillna(fill_value)
                    print(f"Filled {missing_count} missing values in '{col}' with mode: {fill_value}")
                
                elif fill_missing == 'drop':
                    df_clean = df_clean.dropna(subset=[col])
                    print(f"Dropped rows with missing values in '{col}'")
                
                else:
                    print(f"Column '{col}' has {missing_count} missing values, but no filling method applied")
    
    # Reset index after cleaning
    df_clean = df_clean.reset_index(drop=True)
    
    print(f"Cleaning complete. Original shape: {df.shape}, Cleaned shape: {df_clean.shape}")
    return df_clean

def validate_dataframe(df, required_columns=None, numeric_columns=None):
    """
    Validate DataFrame structure and data types.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): Columns that must be present
    numeric_columns (list): Columns that should be numeric
    
    Returns:
    dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Check required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing required columns: {missing_columns}")
    
    # Check numeric columns
    if numeric_columns:
        for col in numeric_columns:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    validation_results['warnings'].append(f"Column '{col}' is not numeric")
    
    # Check for infinite values
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        infinite_count = np.isinf(numeric_df).sum().sum()
        if infinite_count > 0:
            validation_results['warnings'].append(f"Found {infinite_count} infinite values in numeric columns")
    
    return validation_results

# Example usage
if __name__ == "__main__":
    # Create sample data with issues
    data = {
        'id': [1, 2, 3, 4, 5, 5, 6],
        'name': ['Alice', 'Bob', 'Charlie', None, 'Eve', 'Eve', 'Frank'],
        'age': [25, 30, None, 35, 40, 40, 45],
        'score': [85.5, 92.0, 78.5, None, 88.0, 88.0, 95.5]
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned_df = clean_dataset(df, fill_missing='mean', remove_duplicates=True)
    
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    # Validate the cleaned data
    validation = validate_dataframe(
        cleaned_df,
        required_columns=['id', 'name', 'age', 'score'],
        numeric_columns=['age', 'score']
    )
    
    print("\nValidation Results:")
    print(f"Is valid: {validation['is_valid']}")
    print(f"Errors: {validation['errors']}")
    print(f"Warnings: {validation['warnings']}")
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

def calculate_basic_stats(df, column):
    """
    Calculate basic statistics for a column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name
    
    Returns:
    dict: Dictionary containing statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to clean. If None, clean all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{col}'")
    
    return cleaned_df

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    data = {
        'id': range(100),
        'value': np.random.normal(100, 15, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    # Introduce some outliers
    data['value'][0] = 500
    data['value'][1] = -200
    data['value'][2] = 300
    
    df = pd.DataFrame(data)
    
    print("Original DataFrame shape:", df.shape)
    print("\nOriginal statistics:")
    print(calculate_basic_stats(df, 'value'))
    
    cleaned_df = clean_numeric_data(df, ['value'])
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("\nCleaned statistics:")
    print(calculate_basic_stats(cleaned_df, 'value'))
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas DataFrame column using the IQR method.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed from the specified column.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data