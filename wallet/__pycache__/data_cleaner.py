import pandas as pd

def clean_dataframe(df):
    """
    Remove rows with null values and standardize column names.
    
    Args:
        df (pd.DataFrame): Input DataFrame
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    # Remove rows with any null values
    df_cleaned = df.dropna()
    
    # Standardize column names: lowercase and replace spaces with underscores
    df_cleaned.columns = (
        df_cleaned.columns
        .str.lower()
        .str.replace(' ', '_')
        .str.replace(r'[^a-z0-9_]', '', regex=True)
    )
    
    return df_cleaned

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
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

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'Product Name': ['Widget A', 'Widget B', None, 'Widget C'],
        'Price': [10.99, 15.49, 20.99, 12.99],
        'In Stock': [True, False, True, None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned_df = clean_dataframe(df)
    print(cleaned_df)
    
    is_valid, message = validate_dataframe(cleaned_df, ['product_name', 'price'])
    print(f"\nValidation: {is_valid} - {message}")
import pandas as pd

def clean_dataset(df, remove_duplicates=True, fill_missing=None):
    """
    Clean a pandas DataFrame by handling missing values and duplicates.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        remove_duplicates (bool): Whether to remove duplicate rows
        fill_missing (str or dict): Strategy to fill missing values:
            - 'mean': Fill with column mean (numeric only)
            - 'median': Fill with column median (numeric only)
            - 'mode': Fill with column mode
            - dict: Column-specific fill values
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if fill_missing is not None:
        if fill_missing == 'mean':
            cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
        elif fill_missing == 'median':
            cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
        elif fill_missing == 'mode':
            cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
        elif isinstance(fill_missing, dict):
            cleaned_df = cleaned_df.fillna(fill_missing)
    
    # Remove duplicates
    if remove_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    return cleaned_df

def validate_data(df, required_columns=None, numeric_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
        numeric_columns (list): List of columns that should be numeric
    
    Returns:
        dict: Validation results with status and messages
    """
    validation_result = {
        'is_valid': True,
        'messages': [],
        'missing_columns': [],
        'non_numeric_columns': []
    }
    
    # Check required columns
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            validation_result['is_valid'] = False
            validation_result['missing_columns'] = missing
            validation_result['messages'].append(f"Missing required columns: {missing}")
    
    # Check numeric columns
    if numeric_columns:
        non_numeric = []
        for col in numeric_columns:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    non_numeric.append(col)
        
        if non_numeric:
            validation_result['is_valid'] = False
            validation_result['non_numeric_columns'] = non_numeric
            validation_result['messages'].append(f"Non-numeric columns found: {non_numeric}")
    
    return validation_result

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'value': [10.5, 20.3, None, 40.1, 50.0, 50.0],
        'category': ['A', 'B', 'A', None, 'C', 'C']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nMissing values:", df.isnull().sum().to_dict())
    
    # Clean the data
    cleaned = clean_dataset(df, fill_missing={'value': df['value'].mean(), 'category': 'Unknown'})
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    # Validate the cleaned data
    validation = validate_data(
        cleaned, 
        required_columns=['id', 'value', 'category'],
        numeric_columns=['id', 'value']
    )
    print("\nValidation result:", validation)import pandas as pd
import re

def clean_dataframe(df, columns_to_clean=None, remove_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Args:
        df: pandas DataFrame to clean.
        columns_to_clean: List of column names to apply text normalization.
                         If None, all object dtype columns are cleaned.
        remove_duplicates: Boolean indicating whether to remove duplicate rows.
        normalize_text: Boolean indicating whether to normalize text in specified columns.
    
    Returns:
        Cleaned pandas DataFrame.
    """
    cleaned_df = df.copy()
    
    if remove_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if normalize_text and not cleaned_df.empty:
        if columns_to_clean is None:
            text_columns = cleaned_df.select_dtypes(include=['object']).columns
        else:
            text_columns = [col for col in columns_to_clean if col in cleaned_df.columns]
        
        for col in text_columns:
            if cleaned_df[col].dtype == 'object':
                cleaned_df[col] = cleaned_df[col].apply(_normalize_string)
                print(f"Normalized text in column: {col}")
    
    return cleaned_df

def _normalize_string(text):
    """
    Normalize a string by converting to lowercase, removing extra whitespace,
    and standardizing common patterns.
    
    Args:
        text: Input string to normalize.
    
    Returns:
        Normalized string.
    """
    if not isinstance(text, str):
        return text
    
    # Convert to lowercase
    normalized = text.lower()
    
    # Remove leading/trailing whitespace
    normalized = normalized.strip()
    
    # Replace multiple spaces with single space
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # Remove special characters except basic punctuation
    normalized = re.sub(r'[^\w\s.,!?-]', '', normalized)
    
    return normalized

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column.
    
    Args:
        df: pandas DataFrame containing email column.
        email_column: Name of the column containing email addresses.
    
    Returns:
        DataFrame with additional 'email_valid' boolean column.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    validated_df = df.copy()
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    validated_df['email_valid'] = validated_df[email_column].apply(
        lambda x: bool(re.match(email_pattern, str(x))) if pd.notnull(x) else False
    )
    
    valid_count = validated_df['email_valid'].sum()
    total_count = len(validated_df)
    print(f"Valid emails: {valid_count}/{total_count} ({valid_count/total_count*100:.1f}%)")
    
    return validated_df

# Example usage (commented out for production)
# if __name__ == "__main__":
#     # Create sample data
#     data = {
#         'name': ['John Doe', 'Jane Smith', 'John Doe', 'Bob Johnson  '],
#         'email': ['john@example.com', 'jane@example', 'john@example.com', 'bob@example.com'],
#         'notes': ['Important client', '  Regular customer  ', 'Important client', 'New lead']
#     }
#     
#     df = pd.DataFrame(data)
#     print("Original DataFrame:")
#     print(df)
#     print("\nCleaned DataFrame:")
#     cleaned = clean_dataframe(df, columns_to_clean=['name', 'notes'])
#     print(cleaned)
#     
#     print("\nValidated emails:")
#     validated = validate_email_column(cleaned, 'email')
#     print(validated[['email', 'email_valid']])
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(file_path, numeric_columns):
    df = pd.read_csv(file_path)
    
    for column in numeric_columns:
        if column in df.columns:
            df = remove_outliers_iqr(df, column)
            df = normalize_minmax(df, column)
    
    df.to_csv('cleaned_data.csv', index=False)
    return df

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'age': [25, 30, 35, 200, 28, 32, 150, 29],
        'salary': [50000, 60000, 70000, 1000000, 55000, 65000, 800000, 58000]
    })
    
    sample_data.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_dataset('sample_data.csv', ['age', 'salary'])
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")
    print("Cleaning completed successfully")import pandas as pd
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

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a DataFrame column.
    
    Args:
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
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): List of columns to clean. If None, cleans all numeric columns.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for column in columns:
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, column)
            except Exception as e:
                print(f"Warning: Could not process column '{column}': {e}")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list, optional): List of required column names
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"

def example_usage():
    """
    Example usage of the data cleaning functions.
    """
    np.random.seed(42)
    
    data = {
        'id': range(100),
        'value': np.random.normal(100, 15, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(data)
    
    print("Original DataFrame shape:", df.shape)
    
    is_valid, message = validate_dataframe(df, ['id', 'value'])
    print(f"Validation: {message}")
    
    stats = calculate_summary_statistics(df, 'value')
    print("\nOriginal statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")
    
    cleaned_df = clean_numeric_data(df, ['value'])
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    
    cleaned_stats = calculate_summary_statistics(cleaned_df, 'value')
    print("\nCleaned statistics:")
    for key, value in cleaned_stats.items():
        print(f"{key}: {value:.2f}")

if __name__ == "__main__":
    example_usage()