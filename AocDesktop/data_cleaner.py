import numpy as np
import pandas as pd

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
    if max_val - min_val == 0:
        return df[column]
    return (df[column] - min_val) / (max_val - min_val)

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
    return cleaned_df.dropna()

def calculate_statistics(df):
    stats = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        stats[col] = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'median': df[col].median()
        }
    return stats

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': np.random.normal(100, 15, 200),
        'B': np.random.exponential(50, 200),
        'C': np.random.randint(1, 1000, 200)
    })
    
    print("Original dataset shape:", sample_data.shape)
    cleaned = clean_dataset(sample_data, ['A', 'B', 'C'])
    print("Cleaned dataset shape:", cleaned.shape)
    
    stats = calculate_statistics(cleaned)
    for col, values in stats.items():
        print(f"{col}: {values}")import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column):
    """
    Remove outliers using IQR method.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using min-max scaling.
    """
    min_val = data[column].min()
    max_val = data[column].max()
    if max_val - min_val == 0:
        return data[column]
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using z-score.
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    if std_val == 0:
        return data[column]
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(df, numeric_columns):
    """
    Main cleaning function: remove outliers and normalize numeric columns.
    """
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
    return cleaned_df.reset_index(drop=True)

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(50, 15, 100),
        'feature2': np.random.exponential(10, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })
    print("Original data shape:", sample_data.shape)
    cleaned = clean_dataset(sample_data, ['feature1', 'feature2'])
    print("Cleaned data shape:", cleaned.shape)
    print("Cleaned data preview:")
    print(cleaned.head())
import pandas as pd
import numpy as np

def clean_dataset(df, drop_threshold=0.5, fill_strategy='mean'):
    """
    Clean a pandas DataFrame by handling missing values and standardizing column names.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    drop_threshold (float): Threshold for dropping columns with missing values (0 to 1)
    fill_strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', 'zero')
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Standardize column names
    cleaned_df.columns = cleaned_df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # Calculate missing percentage for each column
    missing_percentage = cleaned_df.isnull().sum() / len(cleaned_df)
    
    # Drop columns with missing values above threshold
    columns_to_drop = missing_percentage[missing_percentage > drop_threshold].index
    cleaned_df = cleaned_df.drop(columns=columns_to_drop)
    
    # Fill remaining missing values based on strategy
    for column in cleaned_df.columns:
        if cleaned_df[column].isnull().any():
            if fill_strategy == 'mean' and pd.api.types.is_numeric_dtype(cleaned_df[column]):
                cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mean())
            elif fill_strategy == 'median' and pd.api.types.is_numeric_dtype(cleaned_df[column]):
                cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].median())
            elif fill_strategy == 'mode':
                cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mode()[0])
            elif fill_strategy == 'zero':
                cleaned_df[column] = cleaned_df[column].fillna(0)
            else:
                # For categorical columns or unknown strategy, fill with most frequent value
                cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mode()[0])
    
    # Remove duplicate rows
    cleaned_df = cleaned_df.drop_duplicates()
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'summary': {}
    }
    
    # Check if input is a DataFrame
    if not isinstance(df, pd.DataFrame):
        validation_results['is_valid'] = False
        validation_results['errors'].append('Input is not a pandas DataFrame')
        return validation_results
    
    # Check for empty DataFrame
    if df.empty:
        validation_results['warnings'].append('DataFrame is empty')
    
    # Check required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f'Missing required columns: {missing_columns}')
    
    # Calculate summary statistics
    validation_results['summary'] = {
        'rows': len(df),
        'columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'column_types': df.dtypes.to_dict()
    }
    
    return validation_results

def sample_usage():
    """Demonstrate usage of the data cleaning functions."""
    
    # Create sample data
    sample_data = {
        'Customer ID': [1, 2, 3, 4, 5, None],
        'Order Value': [100.50, None, 75.25, 200.00, 150.75, 90.00],
        'Product Category': ['A', 'B', 'A', None, 'C', 'B'],
        'Discount Applied': [True, False, True, True, None, False]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Validate the data
    validation = validate_dataframe(df, required_columns=['Customer ID', 'Order Value'])
    print("Validation Results:")
    print(validation)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned_df = clean_dataset(df, drop_threshold=0.3, fill_strategy='mean')
    print("Cleaned DataFrame:")
    print(cleaned_df)
    
    return cleaned_df

if __name__ == "__main__":
    cleaned_data = sample_usage()