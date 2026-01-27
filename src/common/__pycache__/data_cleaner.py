import pandas as pd
import numpy as np

def clean_dataset(df, duplicate_threshold=0.8):
    """
    Clean dataset by removing duplicates and handling missing values.
    
    Args:
        df: pandas DataFrame to clean
        duplicate_threshold: similarity threshold for duplicate detection (0.0-1.0)
    
    Returns:
        Cleaned pandas DataFrame
    """
    # Create a copy to avoid modifying original
    cleaned_df = df.copy()
    
    # Remove exact duplicates
    initial_count = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates()
    exact_duplicates = initial_count - len(cleaned_df)
    
    # Remove approximate duplicates based on threshold
    if duplicate_threshold < 1.0:
        similarity_matrix = calculate_similarity(cleaned_df)
        duplicate_mask = identify_approximate_duplicates(similarity_matrix, duplicate_threshold)
        cleaned_df = cleaned_df[~duplicate_mask]
    
    # Handle missing values
    cleaned_df = handle_missing_values(cleaned_df)
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def calculate_similarity(df):
    """
    Calculate similarity matrix for the DataFrame.
    For simplicity, using simple string similarity for text columns.
    """
    # This is a simplified version - in practice you'd use more sophisticated methods
    similarity_scores = np.zeros((len(df), len(df)))
    
    # For demonstration, using a simple approach
    for i in range(len(df)):
        for j in range(i+1, len(df)):
            # Simple similarity calculation (placeholder)
            similarity = np.random.random()  # Replace with actual similarity calculation
            similarity_scores[i, j] = similarity
            similarity_scores[j, i] = similarity
    
    return similarity_scores

def identify_approximate_duplicates(similarity_matrix, threshold):
    """
    Identify approximate duplicates based on similarity threshold.
    """
    n = similarity_matrix.shape[0]
    duplicate_mask = np.zeros(n, dtype=bool)
    
    for i in range(n):
        if not duplicate_mask[i]:
            similar_indices = np.where(similarity_matrix[i] > threshold)[0]
            similar_indices = similar_indices[similar_indices > i]
            duplicate_mask[similar_indices] = True
    
    return duplicate_mask

def handle_missing_values(df):
    """
    Handle missing values in the DataFrame.
    """
    # For numeric columns, fill with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    
    # For categorical columns, fill with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
    
    return df

def validate_data(df):
    """
    Validate data after cleaning.
    """
    validation_results = {
        'total_rows': len(df),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': len(df.select_dtypes(include=['object']).columns)
    }
    
    return validation_results

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'id': [1, 2, 3, 4, 5, 6, 7, 8],
        'name': ['Alice', 'Bob', 'Alice', 'Charlie', 'David', 'Eve', 'Frank', 'Grace'],
        'age': [25, 30, 25, 35, 40, None, 45, 50],
        'score': [85.5, 90.0, 85.5, 92.5, 88.0, 91.0, None, 95.0],
        'department': ['HR', 'IT', 'HR', 'Finance', 'IT', 'Marketing', 'IT', 'HR']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\nDataset info:")
    print(df.info())
    
    # Clean the dataset
    cleaned_df = clean_dataset(df, duplicate_threshold=0.9)
    
    print("\nCleaned dataset:")
    print(cleaned_df)
    
    # Validate the cleaned data
    validation = validate_data(cleaned_df)
    print("\nValidation results:")
    for key, value in validation.items():
        print(f"{key}: {value}")