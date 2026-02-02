
import pandas as pd
import re

def clean_dataframe(df, column_name):
    """
    Clean a specific column in a pandas DataFrame by removing duplicates,
    stripping whitespace, and converting to lowercase.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")

    df[column_name] = df[column_name].astype(str)
    df[column_name] = df[column_name].str.strip()
    df[column_name] = df[column_name].str.lower()
    df.drop_duplicates(subset=[column_name], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def normalize_string(text):
    """
    Normalize a string by removing extra spaces and special characters.
    """
    if not isinstance(text, str):
        return text
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def process_file(input_path, output_path, column_to_clean):
    """
    Read a CSV file, clean the specified column, and save to a new file.
    """
    try:
        df = pd.read_csv(input_path)
        df = clean_dataframe(df, column_to_clean)
        df[column_to_clean] = df[column_to_clean].apply(normalize_string)
        df.to_csv(output_path, index=False)
        print(f"Data cleaned and saved to {output_path}")
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    target_column = "product_name"
    process_file(input_file, output_file, target_column)
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using min-max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using z-score normalization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def handle_missing_values(data, strategy='mean'):
    """
    Handle missing values in numerical columns
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    if strategy == 'mean':
        for col in numeric_cols:
            if data[col].isnull().any():
                data[col] = data[col].fillna(data[col].mean())
    elif strategy == 'median':
        for col in numeric_cols:
            if data[col].isnull().any():
                data[col] = data[col].fillna(data[col].median())
    elif strategy == 'mode':
        for col in numeric_cols:
            if data[col].isnull().any():
                data[col] = data[col].fillna(data[col].mode()[0])
    
    return data

def create_sample_data():
    """
    Create sample data for testing
    """
    np.random.seed(42)
    data = {
        'feature1': np.random.normal(100, 15, 100),
        'feature2': np.random.uniform(0, 1, 100),
        'feature3': np.random.exponential(2, 100)
    }
    
    df = pd.DataFrame(data)
    
    df.loc[np.random.choice(df.index, size=5), 'feature1'] = np.nan
    df.loc[np.random.choice(df.index, size=3), 'feature2'] = np.nan
    
    return df

if __name__ == "__main__":
    sample_data = create_sample_data()
    print("Original data shape:", sample_data.shape)
    print("\nMissing values per column:")
    print(sample_data.isnull().sum())
    
    cleaned_data = handle_missing_values(sample_data.copy(), strategy='mean')
    print("\nAfter handling missing values:")
    print(cleaned_data.isnull().sum())
    
    filtered_data = remove_outliers_iqr(cleaned_data, 'feature1')
    print("\nAfter outlier removal shape:", filtered_data.shape)
    
    normalized_feature = normalize_minmax(filtered_data, 'feature2')
    print("\nNormalized feature2 - first 5 values:")
    print(normalized_feature.head())
    
    standardized_feature = standardize_zscore(filtered_data, 'feature3')
    print("\nStandardized feature3 - first 5 values:")
    print(standardized_feature.head())