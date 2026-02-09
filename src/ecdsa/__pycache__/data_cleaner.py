
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
import pandas as pd
import numpy as np

def clean_dataset(df, column_mapping=None, drop_duplicates=True, fill_na=True):
    """
    Clean a pandas DataFrame by standardizing columns, removing duplicates,
    and handling missing values.
    """
    df_clean = df.copy()
    
    if column_mapping:
        df_clean = df_clean.rename(columns=column_mapping)
    
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates()
    
    if fill_na:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
        
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        df_clean[categorical_cols] = df_clean[categorical_cols].fillna('Unknown')
    
    df_clean = df_clean.reset_index(drop=True)
    return df_clean

def validate_data(df, required_columns=None, unique_constraints=None):
    """
    Validate data integrity by checking required columns and unique constraints.
    """
    validation_results = {}
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        validation_results['missing_columns'] = missing_columns
    
    if unique_constraints:
        duplicate_counts = {}
        for constraint in unique_constraints:
            if constraint in df.columns:
                duplicates = df[constraint].duplicated().sum()
                duplicate_counts[constraint] = duplicates
        validation_results['duplicate_counts'] = duplicate_counts
    
    validation_results['total_rows'] = len(df)
    validation_results['total_columns'] = len(df.columns)
    
    return validation_results

def standardize_text_columns(df, text_columns=None):
    """
    Standardize text columns by converting to lowercase and stripping whitespace.
    """
    df_standardized = df.copy()
    
    if text_columns is None:
        text_columns = df.select_dtypes(include=['object']).columns
    
    for col in text_columns:
        if col in df_standardized.columns:
            df_standardized[col] = df_standardized[col].astype(str).str.lower().str.strip()
    
    return df_standardized

if __name__ == "__main__":
    sample_data = {
        'Name': ['Alice', 'Bob', 'Alice', 'Charlie', None],
        'Age': [25, 30, 25, 35, 40],
        'City': ['New York', 'Los Angeles', 'new york', 'Chicago', None],
        'Salary': [50000, 60000, 50000, 70000, 80000]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned_df = clean_dataset(df)
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\n" + "="*50 + "\n")
    
    validation = validate_data(cleaned_df, required_columns=['Name', 'Age'])
    print("Validation Results:")
    print(validation)