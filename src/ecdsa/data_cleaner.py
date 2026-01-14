
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    return cleaned_df.reset_index(drop=True)

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    })
    cleaned = clean_dataset(sample_data, ['A', 'B', 'C'])
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {cleaned.shape}")import pandas as pd
import numpy as np
from pathlib import Path

def clean_csv_data(input_path, output_path=None, fill_strategy='mean'):
    """
    Clean CSV data by handling missing values and standardizing column names.
    
    Args:
        input_path (str): Path to input CSV file
        output_path (str, optional): Path for cleaned output CSV. 
                                     If None, adds '_cleaned' suffix to input filename.
        fill_strategy (str): Strategy for filling missing values. 
                             Options: 'mean', 'median', 'zero', 'drop'
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    
    # Validate input file exists
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Read CSV file
    df = pd.read_csv(input_path)
    
    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # Handle missing values based on strategy
    if fill_strategy == 'drop':
        df = df.dropna()
    elif fill_strategy == 'zero':
        df = df.fillna(0)
    elif fill_strategy == 'mean':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].mean())
    elif fill_strategy == 'median':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
    else:
        raise ValueError(f"Unknown fill strategy: {fill_strategy}")
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Determine output path
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_cleaned.csv"
    else:
        output_path = Path(output_path)
    
    # Save cleaned data
    df.to_csv(output_path, index=False)
    
    print(f"Data cleaning completed. Cleaned data saved to: {output_path}")
    print(f"Original shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    return df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list, optional): List of required column names
    
    Returns:
        dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'issues': [],
        'summary': {}
    }
    
    # Check if DataFrame is empty
    if df.empty:
        validation_results['is_valid'] = False
        validation_results['issues'].append('DataFrame is empty')
    
    # Check required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f'Missing required columns: {missing_columns}')
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.any(np.isinf(df[col])):
            validation_results['issues'].append(f'Column {col} contains infinite values')
    
    # Generate summary statistics
    validation_results['summary'] = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'numeric_columns': len(numeric_cols),
        'categorical_columns': len(df.select_dtypes(include=['object']).columns),
        'missing_values': df.isnull().sum().sum()
    }
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'Name': ['Alice', 'Bob', 'Charlie', None],
        'Age': [25, 30, None, 35],
        'Salary': [50000, 60000, 55000, None],
        'Department': ['HR', 'IT', 'IT', 'Finance']
    }
    
    df = pd.DataFrame(sample_data)
    test_file = Path('test_data.csv')
    df.to_csv(test_file, index=False)
    
    try:
        cleaned_df = clean_csv_data('test_data.csv', fill_strategy='mean')
        validation = validate_dataframe(cleaned_df, required_columns=['name', 'age', 'salary'])
        print(f"Validation passed: {validation['is_valid']}")
        print(f"Validation summary: {validation['summary']}")
    finally:
        # Clean up test file
        if test_file.exists():
            test_file.unlink()
        output_file = Path('test_data_cleaned.csv')
        if output_file.exists():
            output_file.unlink()import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        subset (list, optional): Columns to consider for identifying duplicates.
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    return df.drop_duplicates(subset=subset, keep='first')

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        strategy (str): Strategy for imputation ('mean', 'median', 'mode', 'drop').
        columns (list, optional): Specific columns to apply imputation.
    
    Returns:
        pd.DataFrame: DataFrame with missing values handled.
    """
    df_copy = df.copy()
    
    if columns is None:
        columns = df_copy.columns
    
    for col in columns:
        if df_copy[col].dtype in [np.float64, np.int64]:
            if strategy == 'mean':
                df_copy[col].fillna(df_copy[col].mean(), inplace=True)
            elif strategy == 'median':
                df_copy[col].fillna(df_copy[col].median(), inplace=True)
            elif strategy == 'mode':
                df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)
        elif strategy == 'drop':
            df_copy = df_copy.dropna(subset=[col])
    
    return df_copy

def normalize_column(df, column, method='minmax'):
    """
    Normalize a column in the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name to normalize.
        method (str): Normalization method ('minmax' or 'zscore').
    
    Returns:
        pd.DataFrame: DataFrame with normalized column.
    """
    df_copy = df.copy()
    
    if method == 'minmax':
        min_val = df_copy[column].min()
        max_val = df_copy[column].max()
        if max_val != min_val:
            df_copy[column] = (df_copy[column] - min_val) / (max_val - min_val)
    elif method == 'zscore':
        mean_val = df_copy[column].mean()
        std_val = df_copy[column].std()
        if std_val != 0:
            df_copy[column] = (df_copy[column] - mean_val) / std_val
    
    return df_copy

def filter_outliers(df, column, method='iqr', threshold=1.5):
    """
    Filter outliers from a DataFrame column.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name to filter.
        method (str): Outlier detection method ('iqr' or 'zscore').
        threshold (float): Threshold for outlier detection.
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    df_copy = df.copy()
    
    if method == 'iqr':
        Q1 = df_copy[column].quantile(0.25)
        Q3 = df_copy[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        df_copy = df_copy[(df_copy[column] >= lower_bound) & (df_copy[column] <= upper_bound)]
    elif method == 'zscore':
        z_scores = np.abs((df_copy[column] - df_copy[column].mean()) / df_copy[column].std())
        df_copy = df_copy[z_scores < threshold]
    
    return df_copy

def clean_dataframe(df, operations):
    """
    Apply multiple cleaning operations to a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        operations (list): List of cleaning operations to apply.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    for operation in operations:
        if operation['type'] == 'remove_duplicates':
            cleaned_df = remove_duplicates(cleaned_df, subset=operation.get('subset'))
        elif operation['type'] == 'handle_missing':
            cleaned_df = handle_missing_values(
                cleaned_df, 
                strategy=operation.get('strategy', 'mean'),
                columns=operation.get('columns')
            )
        elif operation['type'] == 'normalize':
            cleaned_df = normalize_column(
                cleaned_df,
                column=operation['column'],
                method=operation.get('method', 'minmax')
            )
        elif operation['type'] == 'filter_outliers':
            cleaned_df = filter_outliers(
                cleaned_df,
                column=operation['column'],
                method=operation.get('method', 'iqr'),
                threshold=operation.get('threshold', 1.5)
            )
    
    return cleaned_df