import pandas as pd

def clean_dataset(df):
    """
    Remove duplicate rows and fill missing numeric values with column mean.
    """
    # Remove duplicates
    df_cleaned = df.drop_duplicates()
    
    # Fill missing numeric values
    numeric_cols = df_cleaned.select_dtypes(include=['number']).columns
    df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].mean())
    
    return df_cleaned

def main():
    # Example usage
    data = {
        'A': [1, 2, None, 4, 1],
        'B': [5, None, 7, 8, 5],
        'C': ['x', 'y', 'z', 'x', 'x']
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_dataset(df)
    print("\nCleaned DataFrame:")
    print(cleaned_df)

if __name__ == "__main__":
    main()import csv
import os

def load_csv(file_path):
    """Load CSV file and return data as list of dictionaries."""
    data = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return data

def clean_numeric_fields(data, fields):
    """Remove non-numeric characters from specified fields."""
    cleaned_data = []
    for row in data:
        cleaned_row = row.copy()
        for field in fields:
            if field in cleaned_row:
                value = cleaned_row[field]
                if isinstance(value, str):
                    cleaned_value = ''.join(char for char in value if char.isdigit() or char == '.')
                    cleaned_row[field] = cleaned_value if cleaned_value else '0'
        cleaned_data.append(cleaned_row)
    return cleaned_data

def remove_empty_rows(data, required_fields):
    """Remove rows where required fields are empty."""
    return [row for row in data if all(row.get(field) for field in required_fields)]

def save_cleaned_csv(data, output_path):
    """Save cleaned data to a new CSV file."""
    if not data:
        raise ValueError("No data to save")
    
    fieldnames = data[0].keys()
    with open(output_path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    return output_path

def process_csv(input_file, output_file, numeric_fields=None, required_fields=None):
    """Main function to process and clean CSV data."""
    if numeric_fields is None:
        numeric_fields = []
    if required_fields is None:
        required_fields = []
    
    try:
        data = load_csv(input_file)
        data = clean_numeric_fields(data, numeric_fields)
        data = remove_empty_rows(data, required_fields)
        save_cleaned_csv(data, output_file)
        return True, f"Data cleaned successfully. Saved to: {output_file}"
    except Exception as e:
        return False, f"Error processing file: {str(e)}"
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to drop duplicate rows
        fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
        print("Dropped rows with missing values")
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        print(f"Filled missing numeric values with {fill_missing}")
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
        print("Filled missing values with mode")
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
        min_rows (int): Minimum number of rows required
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Data validation passed"

def remove_outliers(df, column, method='iqr', threshold=1.5):
    """
    Remove outliers from a specific column using specified method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to process
        method (str): Outlier detection method ('iqr' or 'zscore')
        threshold (float): Threshold for outlier detection
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' must be numeric")
    
    data = df[column].copy()
    
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        mask = (data >= lower_bound) & (data <= upper_bound)
    elif method == 'zscore':
        z_scores = np.abs((data - data.mean()) / data.std())
        mask = z_scores <= threshold
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")
    
    removed_count = len(data) - mask.sum()
    print(f"Removed {removed_count} outliers from column '{column}'")
    
    return df[mask].reset_index(drop=True)
import pandas as pd
import numpy as np

def clean_dataframe(df):
    """
    Clean a pandas DataFrame by removing duplicates and standardizing column names.
    """
    # Create a copy to avoid modifying the original DataFrame
    cleaned_df = df.copy()
    
    # Remove duplicate rows
    cleaned_df = cleaned_df.drop_duplicates()
    
    # Standardize column names: lowercase and replace spaces with underscores
    cleaned_df.columns = cleaned_df.columns.str.lower().str.replace(' ', '_')
    
    # Fill missing numeric values with column mean
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].mean())
    
    # Fill missing categorical values with mode
    categorical_cols = cleaned_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        mode_value = cleaned_df[col].mode()
        if not mode_value.empty:
            cleaned_df[col] = cleaned_df[col].fillna(mode_value.iloc[0])
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    """
    if required_columns:
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    return True

def main():
    # Example usage
    sample_data = {
        'Name': ['Alice', 'Bob', 'Alice', 'Charlie', None],
        'Age': [25, 30, 25, 35, 40],
        'City': ['New York', 'London', 'New York', 'Paris', 'Tokyo'],
        'Salary': [50000, 60000, 50000, 70000, None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    cleaned_df = clean_dataframe(df)
    print("Cleaned DataFrame:")
    print(cleaned_df)
    
    try:
        validate_dataframe(cleaned_df, required_columns=['name', 'age', 'city'])
        print("\nData validation passed")
    except ValueError as e:
        print(f"\nData validation failed: {e}")

if __name__ == "__main__":
    main()
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Args:
        data (list or np.array): Input data
        column (int): Column index for 2D data, or None for 1D data
    
    Returns:
        np.array: Data with outliers removed
    """
    if column is not None:
        column_data = data[:, column]
    else:
        column_data = np.array(data)
    
    q1 = np.percentile(column_data, 25)
    q3 = np.percentile(column_data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    if column is not None:
        mask = (data[:, column] >= lower_bound) & (data[:, column] <= upper_bound)
        return data[mask]
    else:
        mask = (column_data >= lower_bound) & (column_data <= upper_bound)
        return column_data[mask]

def calculate_statistics(data):
    """
    Calculate basic statistics for the data.
    
    Args:
        data (np.array): Input data
    
    Returns:
        dict: Dictionary containing mean, median, std
    """
    return {
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data)
    }

def clean_dataset(data, columns_to_clean=None):
    """
    Clean dataset by removing outliers from specified columns.
    
    Args:
        data (np.array): 2D array of data
        columns_to_clean (list): List of column indices to clean
    
    Returns:
        np.array: Cleaned dataset
    """
    if columns_to_clean is None:
        columns_to_clean = list(range(data.shape[1]))
    
    cleaned_data = data.copy()
    
    for col in columns_to_clean:
        if col < data.shape[1]:
            cleaned_data = remove_outliers_iqr(cleaned_data, col)
    
    return cleaned_data

if __name__ == "__main__":
    # Example usage
    sample_data = np.random.randn(100, 3) * 10 + 50
    print(f"Original data shape: {sample_data.shape}")
    
    cleaned = clean_dataset(sample_data, columns_to_clean=[0, 1, 2])
    print(f"Cleaned data shape: {cleaned.shape}")
    
    stats = calculate_statistics(cleaned[:, 0])
    print(f"Statistics for first column: {stats}")
import pandas as pd
import numpy as np
from scipy import stats

def load_data(filepath):
    return pd.read_csv(filepath)

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def remove_outliers_zscore(df, column, threshold=3):
    z_scores = np.abs(stats.zscore(df[column]))
    return df[z_scores < threshold]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def normalize_zscore(df, column):
    mean_val = df[column].mean()
    std_val = df[column].std()
    df[column + '_standardized'] = (df[column] - mean_val) / std_val
    return df

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize_method='minmax'):
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if outlier_method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        elif outlier_method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, col)
        
        if normalize_method == 'minmax':
            cleaned_df = normalize_minmax(cleaned_df, col)
        elif normalize_method == 'zscore':
            cleaned_df = normalize_zscore(cleaned_df, col)
    
    return cleaned_df

def save_cleaned_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    numeric_cols = ['age', 'income', 'score']
    
    raw_data = load_data(input_file)
    cleaned_data = clean_dataset(raw_data, numeric_cols, outlier_method='iqr', normalize_method='zscore')
    save_cleaned_data(cleaned_data, output_file)import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using Z-score normalization.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in specified columns.
    """
    if columns is None:
        columns = data.columns
    
    data_filled = data.copy()
    
    for col in columns:
        if col not in data.columns:
            continue
        
        if strategy == 'mean':
            fill_value = data[col].mean()
        elif strategy == 'median':
            fill_value = data[col].median()
        elif strategy == 'mode':
            fill_value = data[col].mode()[0] if not data[col].mode().empty else 0
        elif strategy == 'constant':
            fill_value = 0
        else:
            raise ValueError("Strategy must be 'mean', 'median', 'mode', or 'constant'")
        
        data_filled[col] = data[col].fillna(fill_value)
    
    return data_filled

def create_sample_data():
    """
    Create sample data for testing.
    """
    np.random.seed(42)
    data = {
        'feature1': np.random.normal(100, 15, 100),
        'feature2': np.random.exponential(50, 100),
        'feature3': np.random.randint(1, 100, 100)
    }
    
    df = pd.DataFrame(data)
    
    indices = np.random.choice(df.index, size=10, replace=False)
    for col in df.columns:
        df.loc[indices[:5], col] = np.nan
    
    return df

if __name__ == "__main__":
    sample_data = create_sample_data()
    print("Original data shape:", sample_data.shape)
    print("Missing values per column:")
    print(sample_data.isnull().sum())
    
    cleaned_data = handle_missing_values(sample_data, strategy='mean')
    print("\nAfter handling missing values:")
    print("Missing values per column:")
    print(cleaned_data.isnull().sum())
    
    for col in cleaned_data.columns:
        cleaned_data = remove_outliers_iqr(cleaned_data, col)
    
    print("\nAfter outlier removal:")
    print("Data shape:", cleaned_data.shape)
    
    for col in cleaned_data.columns:
        cleaned_data[f'{col}_normalized'] = normalize_minmax(cleaned_data, col)
        cleaned_data[f'{col}_standardized'] = standardize_zscore(cleaned_data, col)
    
    print("\nFinal data columns:", cleaned_data.columns.tolist())
    print("\nFirst 5 rows of processed data:")
    print(cleaned_data.head())import pandas as pd
import numpy as np

def clean_csv_data(file_path, output_path=None):
    """
    Load a CSV file, perform basic cleaning operations,
    and save the cleaned data.
    """
    try:
        df = pd.read_csv(file_path)
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        # Fill missing numeric values with column mean
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].mean())
        
        # Fill missing categorical values with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        # Remove leading/trailing whitespace from string columns
        for col in categorical_cols:
            df[col] = df[col].str.strip()
        
        # Save cleaned data
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
        else:
            base_name = file_path.rsplit('.', 1)[0]
            cleaned_path = f"{base_name}_cleaned.csv"
            df.to_csv(cleaned_path, index=False)
            print(f"Cleaned data saved to: {cleaned_path}")
        
        return df
    
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except pd.errors.EmptyDataError:
        print("Error: The file is empty")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df):
    """
    Perform basic validation on a DataFrame.
    """
    if df is None or df.empty:
        print("DataFrame is empty or None")
        return False
    
    print(f"DataFrame shape: {df.shape}")
    print(f"Missing values per column:")
    print(df.isnull().sum())
    print(f"Data types:")
    print(df.dtypes)
    
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'name': ['Alice', 'Bob', 'Charlie', None, 'Eve', 'Eve'],
        'age': [25, 30, None, 35, 40, 40],
        'score': [85.5, 92.0, 78.5, None, 88.0, 88.0],
        'city': ['New York ', ' Los Angeles', 'Chicago', 'Boston', None, 'Boston']
    }
    
    # Create a test DataFrame
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    # Clean the data
    cleaned_df = clean_csv_data('test_data.csv', 'cleaned_test_data.csv')
    
    # Validate the cleaned data
    if cleaned_df is not None:
        validate_dataframe(cleaned_df)
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to drop duplicate rows
        fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop')
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
        print("Dropped rows with missing values")
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                fill_value = cleaned_df[col].mean()
            else:
                fill_value = cleaned_df[col].median()
            cleaned_df[col] = cleaned_df[col].fillna(fill_value)
        print(f"Filled missing numeric values with {fill_missing}")
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            mode_value = cleaned_df[col].mode()
            if not mode_value.empty:
                cleaned_df[col] = cleaned_df[col].fillna(mode_value[0])
        print("Filled missing values with mode")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    if not isinstance(df, pd.DataFrame):
        validation_results['is_valid'] = False
        validation_results['errors'].append("Input is not a pandas DataFrame")
        return validation_results
    
    if df.empty:
        validation_results['warnings'].append("DataFrame is empty")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing required columns: {missing_cols}")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().all():
            validation_results['warnings'].append(f"Column '{col}' contains only NaN values")
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 3, None],
        'B': [4, None, 4, 6, 7],
        'C': ['x', 'y', 'x', 'z', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    validation = validate_dataframe(df, required_columns=['A', 'B'])
    print("Validation Results:")
    print(validation)
    print("\n" + "="*50 + "\n")
    
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("Cleaned DataFrame:")
    print(cleaned)import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop').
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            elif fill_missing == 'median':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame for basic integrity checks.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    tuple: (is_valid, message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, None],
        'B': [5, None, 7, 8, 9],
        'C': ['x', 'y', 'y', 'z', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid, msg = validate_dataframe(cleaned, required_columns=['A', 'B'])
    print(f"\nValidation: {msg}")
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, columns=None):
    """
    Remove outliers from DataFrame using Interquartile Range method.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to process (default: all numeric columns)
    
    Returns:
        DataFrame with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_clean = df.copy()
    
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
            df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def standardize_columns(df, columns=None):
    """
    Standardize numeric columns to have zero mean and unit variance.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to standardize
    
    Returns:
        DataFrame with standardized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_standardized = df.copy()
    
    for col in columns:
        if col in df.columns and df[col].std() > 0:
            df_standardized[col] = (df[col] - df[col].mean()) / df[col].std()
    
    return df_standardized

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in DataFrame.
    
    Args:
        df: pandas DataFrame
        strategy: 'mean', 'median', 'mode', or 'drop'
        columns: list of column names to process
    
    Returns:
        DataFrame with handled missing values
    """
    if columns is None:
        columns = df.columns.tolist()
    
    df_processed = df.copy()
    
    for col in columns:
        if col in df.columns and df[col].isnull().any():
            if strategy == 'mean' and np.issubdtype(df[col].dtype, np.number):
                df_processed[col].fillna(df[col].mean(), inplace=True)
            elif strategy == 'median' and np.issubdtype(df[col].dtype, np.number):
                df_processed[col].fillna(df[col].median(), inplace=True)
            elif strategy == 'mode':
                df_processed[col].fillna(df[col].mode()[0], inplace=True)
            elif strategy == 'drop':
                df_processed = df_processed.dropna(subset=[col])
    
    return df_processed.reset_index(drop=True)import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
        fill_missing (str): Method to fill missing values. 
                           Options: 'mean', 'median', 'mode', or 'drop'. 
                           Default is 'mean'.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median', 'mode']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
            elif fill_missing == 'median':
                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
            elif fill_missing == 'mode':
                cleaned_df[col].fillna(cleaned_df[col].mode()[0], inplace=True)
    
    categorical_cols = cleaned_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        cleaned_df[col].fillna('Unknown', inplace=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate a DataFrame for basic integrity checks.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
    
    Returns:
        tuple: (is_valid, message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

def remove_outliers(df, column, method='iqr', threshold=1.5):
    """
    Remove outliers from a specific column in DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name to process.
        method (str): Outlier detection method. Options: 'iqr' or 'zscore'.
        threshold (float): Threshold for outlier detection.
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' must be numeric")
    
    data = df[column].copy()
    
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        mask = (data >= lower_bound) & (data <= upper_bound)
    elif method == 'zscore':
        z_scores = (data - data.mean()) / data.std()
        mask = abs(z_scores) <= threshold
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")
    
    return df[mask]
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    def remove_outliers_iqr(self, columns=None, multiplier=1.5):
        if columns is None:
            columns = self.numeric_columns
        
        clean_df = self.df.copy()
        for col in columns:
            if col not in self.numeric_columns:
                continue
                
            Q1 = clean_df[col].quantile(0.25)
            Q3 = clean_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            mask = (clean_df[col] >= lower_bound) & (clean_df[col] <= upper_bound)
            clean_df = clean_df[mask]
        
        return clean_df
    
    def remove_outliers_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.numeric_columns
        
        clean_df = self.df.copy()
        for col in columns:
            if col not in self.numeric_columns:
                continue
            
            z_scores = np.abs(stats.zscore(clean_df[col].dropna()))
            mask = z_scores < threshold
            clean_df = clean_df[mask]
        
        return clean_df
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        normalized_df = self.df.copy()
        for col in columns:
            if col not in self.numeric_columns:
                continue
            
            col_min = normalized_df[col].min()
            col_max = normalized_df[col].max()
            
            if col_max - col_min > 0:
                normalized_df[col] = (normalized_df[col] - col_min) / (col_max - col_min)
        
        return normalized_df
    
    def normalize_zscore(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        normalized_df = self.df.copy()
        for col in columns:
            if col not in self.numeric_columns:
                continue
            
            col_mean = normalized_df[col].mean()
            col_std = normalized_df[col].std()
            
            if col_std > 0:
                normalized_df[col] = (normalized_df[col] - col_mean) / col_std
        
        return normalized_df
    
    def fill_missing_mean(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
        
        filled_df = self.df.copy()
        for col in columns:
            if col not in self.numeric_columns:
                continue
            
            col_mean = filled_df[col].mean()
            filled_df[col] = filled_df[col].fillna(col_mean)
        
        return filled_df
    
    def get_summary(self):
        summary = {
            'original_shape': self.df.shape,
            'numeric_columns': self.numeric_columns,
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.to_dict()
        }
        return summary

def example_usage():
    np.random.seed(42)
    data = {
        'feature1': np.random.normal(100, 15, 100),
        'feature2': np.random.exponential(50, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(data)
    df.loc[10:15, 'feature1'] = np.nan
    df.loc[5, 'feature2'] = 1000
    
    cleaner = DataCleaner(df)
    print("Original data shape:", cleaner.get_summary()['original_shape'])
    
    clean_df = cleaner.remove_outliers_iqr(['feature1', 'feature2'])
    print("After IQR outlier removal:", clean_df.shape)
    
    normalized_df = cleaner.normalize_minmax(['feature1', 'feature2'])
    print("After min-max normalization:", normalized_df.shape)
    
    filled_df = cleaner.fill_missing_mean(['feature1'])
    print("After filling missing values:", filled_df.shape)

if __name__ == "__main__":
    example_usage()import pandas as pd
import numpy as np

def clean_dataframe(df, fill_strategy='mean', column_case='lower'):
    """
    Clean a pandas DataFrame by handling missing values and standardizing column names.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    fill_strategy (str): Strategy for filling missing values. Options: 'mean', 'median', 'mode', 'zero', 'drop'.
    column_case (str): Target case for column names. Options: 'lower', 'upper', 'title'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Standardize column names
    if column_case == 'lower':
        cleaned_df.columns = cleaned_df.columns.str.lower()
    elif column_case == 'upper':
        cleaned_df.columns = cleaned_df.columns.str.upper()
    elif column_case == 'title':
        cleaned_df.columns = cleaned_df.columns.str.title()
    
    # Handle missing values
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    categorical_cols = cleaned_df.select_dtypes(exclude=[np.number]).columns
    
    if fill_strategy == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_strategy == 'zero':
        cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(0)
        cleaned_df[categorical_cols] = cleaned_df[categorical_cols].fillna('Unknown')
    elif fill_strategy == 'mean':
        cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].mean())
        cleaned_df[categorical_cols] = cleaned_df[categorical_cols].fillna('Unknown')
    elif fill_strategy == 'median':
        cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].median())
        cleaned_df[categorical_cols] = cleaned_df[categorical_cols].fillna('Unknown')
    elif fill_strategy == 'mode':
        for col in cleaned_df.columns:
            if col in numeric_cols:
                mode_val = cleaned_df[col].mode()
                cleaned_df[col] = cleaned_df[col].fillna(mode_val[0] if not mode_val.empty else 0)
            else:
                mode_val = cleaned_df[col].mode()
                cleaned_df[col] = cleaned_df[col].fillna(mode_val[0] if not mode_val.empty else 'Unknown')
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

# Example usage (commented out for production)
# if __name__ == "__main__":
#     sample_data = {
#         'Name': ['Alice', 'Bob', None, 'David'],
#         'Age': [25, None, 30, 35],
#         'Score': [85.5, 92.0, None, 78.5]
#     }
#     
#     df = pd.DataFrame(sample_data)
#     print("Original DataFrame:")
#     print(df)
#     
#     cleaned = clean_dataframe(df, fill_strategy='mean', column_case='lower')
#     print("\nCleaned DataFrame:")
#     print(cleaned)
#     
#     is_valid, message = validate_dataframe(cleaned, required_columns=['name', 'age'], min_rows=2)
#     print(f"\nValidation: {is_valid} - {message}")
import pandas as pd
import numpy as np
from pathlib import Path

def clean_csv_data(input_path, output_path=None):
    """
    Clean CSV data by handling missing values and removing duplicates.
    """
    try:
        df = pd.read_csv(input_path)
        
        # Remove duplicate rows
        initial_count = len(df)
        df = df.drop_duplicates()
        duplicates_removed = initial_count - len(df)
        
        # Handle missing values
        missing_before = df.isnull().sum().sum()
        
        # Fill numeric columns with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        
        # Fill categorical columns with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        missing_after = df.isnull().sum().sum()
        
        # Generate output path if not provided
        if output_path is None:
            input_file = Path(input_path)
            output_path = input_file.parent / f"cleaned_{input_file.name}"
        
        # Save cleaned data
        df.to_csv(output_path, index=False)
        
        # Print summary
        print(f"Data cleaning completed:")
        print(f"  - Removed {duplicates_removed} duplicate rows")
        print(f"  - Fixed {missing_before - missing_after} missing values")
        print(f"  - Cleaned data saved to: {output_path}")
        print(f"  - Final dataset shape: {df.shape}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_path}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: Input file '{input_path}' is empty.")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate dataframe structure and content.
    """
    if df is None or df.empty:
        print("Error: Dataframe is empty or None.")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    # Check for any remaining null values
    if df.isnull().sum().sum() > 0:
        print(f"Warning: Dataframe still contains {df.isnull().sum().sum()} null values.")
    
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'name': ['Alice', 'Bob', None, 'David', 'Eve', 'Eve'],
        'age': [25, 30, None, 35, 40, 40],
        'score': [85.5, 92.0, 78.5, None, 88.0, 88.0]
    }
    
    # Create sample CSV
    temp_df = pd.DataFrame(sample_data)
    temp_df.to_csv('sample_data.csv', index=False)
    
    # Clean the data
    cleaned_df = clean_csv_data('sample_data.csv', 'cleaned_sample_data.csv')
    
    # Validate the cleaned data
    if cleaned_df is not None:
        is_valid = validate_dataframe(cleaned_df, ['id', 'name', 'age', 'score'])
        if is_valid:
            print("Data validation passed.")
        else:
            print("Data validation failed.")
    
    # Clean up sample files
    import os
    if os.path.exists('sample_data.csv'):
        os.remove('sample_data.csv')
    if os.path.exists('cleaned_sample_data.csv'):
        print(f"Cleaned file preserved: cleaned_sample_data.csv")import pandas as pd
import numpy as np

def clean_csv_data(file_path, strategy='mean', columns=None):
    """
    Load a CSV file and handle missing values in specified columns.
    
    Args:
        file_path (str): Path to the CSV file.
        strategy (str): Strategy for imputation ('mean', 'median', 'mode', 'drop').
        columns (list): List of column names to process. If None, process all numeric columns.
    
    Returns:
        pandas.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    for col in columns:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in DataFrame. Skipping.")
            continue
        
        if df[col].isnull().sum() > 0:
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
            
            df[col].fillna(fill_value, inplace=True)
            print(f"Filled missing values in column '{col}' using {strategy} value: {fill_value}")
    
    return df

def save_cleaned_data(df, output_path):
    """
    Save the cleaned DataFrame to a new CSV file.
    
    Args:
        df (pandas.DataFrame): Cleaned DataFrame.
        output_path (str): Path for the output CSV file.
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = clean_csv_data(input_file, strategy='median')
    save_cleaned_data(cleaned_df, output_file)
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
    
    return filtered_df

def calculate_summary_stats(df, column):
    """
    Calculate summary statistics for a column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name
    
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

def clean_dataset(df, numeric_columns=None):
    """
    Clean a dataset by removing outliers from all numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of numeric column names. If None, uses all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in df.columns:
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, column)
            except Exception as e:
                print(f"Warning: Could not clean column '{column}': {e}")
    
    return cleaned_df

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'id': range(100),
        'value': np.random.normal(100, 15, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })
    
    # Add some outliers
    sample_data.loc[0, 'value'] = 500
    sample_data.loc[1, 'value'] = -200
    
    print("Original dataset shape:", sample_data.shape)
    print("Original stats:", calculate_summary_stats(sample_data, 'value'))
    
    cleaned_data = clean_dataset(sample_data, ['value'])
    
    print("\nCleaned dataset shape:", cleaned_data.shape)
    print("Cleaned stats:", calculate_summary_stats(cleaned_data, 'value'))
import pandas as pd
import numpy as np
from typing import Optional, Union, List

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_duplicates(self, subset: Optional[List[str]] = None, keep: str = 'first') -> 'DataCleaner':
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        return self
        
    def convert_dtypes(self, column_type_map: dict) -> 'DataCleaner':
        for column, dtype in column_type_map.items():
            if column in self.df.columns:
                if dtype == 'datetime':
                    self.df[column] = pd.to_datetime(self.df[column], errors='coerce')
                elif dtype == 'numeric':
                    self.df[column] = pd.to_numeric(self.df[column], errors='coerce')
                elif dtype == 'category':
                    self.df[column] = self.df[column].astype('category')
        return self
        
    def fill_missing(self, strategy: str = 'mean', custom_value: Optional[Union[int, float, str]] = None) -> 'DataCleaner':
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if strategy == 'mean' and len(numeric_cols) > 0:
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
        elif strategy == 'median' and len(numeric_cols) > 0:
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        elif strategy == 'custom' and custom_value is not None:
            self.df = self.df.fillna(custom_value)
        elif strategy == 'ffill':
            self.df = self.df.fillna(method='ffill')
            
        return self
        
    def remove_outliers(self, column: str, method: str = 'iqr', threshold: float = 1.5) -> 'DataCleaner':
        if column not in self.df.columns or not pd.api.types.is_numeric_dtype(self.df[column]):
            return self
            
        if method == 'iqr':
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            self.df = self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]
            
        return self
        
    def get_cleaned_data(self) -> pd.DataFrame:
        return self.df
        
    def get_summary(self) -> dict:
        removed_rows = self.original_shape[0] - self.df.shape[0]
        removed_cols = self.original_shape[1] - self.df.shape[1]
        
        return {
            'original_shape': self.original_shape,
            'cleaned_shape': self.df.shape,
            'removed_rows': removed_rows,
            'removed_cols': removed_cols,
            'missing_values': self.df.isnull().sum().to_dict(),
            'dtypes': self.df.dtypes.to_dict()
        }

def clean_dataset(df: pd.DataFrame, 
                  remove_dups: bool = True,
                  fill_na: bool = True,
                  outlier_columns: Optional[List[str]] = None) -> pd.DataFrame:
    
    cleaner = DataCleaner(df)
    
    if remove_dups:
        cleaner.remove_duplicates()
    
    if fill_na:
        cleaner.fill_missing(strategy='mean')
    
    if outlier_columns:
        for col in outlier_columns:
            if col in df.columns:
                cleaner.remove_outliers(col)
    
    return cleaner.get_cleaned_data()
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
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
        'count': df[column].count()
    }
    
    return stats

def main():
    # Example usage
    np.random.seed(42)
    data = {
        'values': np.concatenate([
            np.random.normal(100, 15, 100),
            np.array([300, 350, -50, 400])  # Outliers
        ])
    }
    
    df = pd.DataFrame(data)
    print(f"Original data shape: {df.shape}")
    print(f"Original statistics: {calculate_summary_statistics(df, 'values')}")
    
    cleaned_df = remove_outliers_iqr(df, 'values')
    print(f"\nCleaned data shape: {cleaned_df.shape}")
    print(f"Cleaned statistics: {calculate_summary_statistics(cleaned_df, 'values')}")

if __name__ == "__main__":
    main()
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, columns, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Parameters:
    dataframe: pandas DataFrame
    columns: list of column names to process
    factor: IQR multiplier (default 1.5)
    
    Returns:
    Cleaned DataFrame with outliers removed
    """
    df_clean = dataframe.copy()
    
    for column in columns:
        if column not in df_clean.columns:
            continue
            
        Q1 = df_clean[column].quantile(0.25)
        Q3 = df_clean[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        mask = (df_clean[column] >= lower_bound) & (df_clean[column] <= upper_bound)
        df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def normalize_minmax(dataframe, columns):
    """
    Normalize columns using Min-Max scaling to [0, 1] range.
    
    Parameters:
    dataframe: pandas DataFrame
    columns: list of column names to normalize
    
    Returns:
    DataFrame with normalized columns
    """
    df_normalized = dataframe.copy()
    
    for column in columns:
        if column not in df_normalized.columns:
            continue
            
        col_min = df_normalized[column].min()
        col_max = df_normalized[column].max()
        
        if col_max != col_min:
            df_normalized[column] = (df_normalized[column] - col_min) / (col_max - col_min)
        else:
            df_normalized[column] = 0
    
    return df_normalized

def standardize_zscore(dataframe, columns):
    """
    Standardize columns using Z-score normalization.
    
    Parameters:
    dataframe: pandas DataFrame
    columns: list of column names to standardize
    
    Returns:
    DataFrame with standardized columns
    """
    df_standardized = dataframe.copy()
    
    for column in columns:
        if column not in df_standardized.columns:
            continue
            
        mean_val = df_standardized[column].mean()
        std_val = df_standardized[column].std()
        
        if std_val > 0:
            df_standardized[column] = (df_standardized[column] - mean_val) / std_val
        else:
            df_standardized[column] = 0
    
    return df_standardized

def handle_missing_values(dataframe, strategy='mean', columns=None):
    """
    Handle missing values in specified columns.
    
    Parameters:
    dataframe: pandas DataFrame
    strategy: 'mean', 'median', 'mode', or 'drop'
    columns: list of column names (if None, process all numeric columns)
    
    Returns:
    DataFrame with handled missing values
    """
    df_processed = dataframe.copy()
    
    if columns is None:
        columns = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    
    for column in columns:
        if column not in df_processed.columns:
            continue
            
        if strategy == 'drop':
            df_processed = df_processed.dropna(subset=[column])
        elif strategy == 'mean':
            df_processed[column] = df_processed[column].fillna(df_processed[column].mean())
        elif strategy == 'median':
            df_processed[column] = df_processed[column].fillna(df_processed[column].median())
        elif strategy == 'mode':
            mode_val = df_processed[column].mode()
            if not mode_val.empty:
                df_processed[column] = df_processed[column].fillna(mode_val.iloc[0])
    
    return df_processed.reset_index(drop=True)

def clean_dataset(dataframe, numeric_columns=None, outlier_factor=1.5, 
                  normalize=False, standardize=False, missing_strategy='mean'):
    """
    Complete data cleaning pipeline.
    
    Parameters:
    dataframe: pandas DataFrame
    numeric_columns: list of numeric column names to process
    outlier_factor: IQR multiplier for outlier removal
    normalize: whether to apply min-max normalization
    standardize: whether to apply z-score standardization
    missing_strategy: strategy for handling missing values
    
    Returns:
    Cleaned and processed DataFrame
    """
    if numeric_columns is None:
        numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    df_clean = dataframe.copy()
    
    df_clean = handle_missing_values(df_clean, strategy=missing_strategy, columns=numeric_columns)
    
    df_clean = remove_outliers_iqr(df_clean, columns=numeric_columns, factor=outlier_factor)
    
    if normalize:
        df_clean = normalize_minmax(df_clean, columns=numeric_columns)
    
    if standardize:
        df_clean = standardize_zscore(df_clean, columns=numeric_columns)
    
    return df_clean

def validate_dataframe(dataframe, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    dataframe: pandas DataFrame to validate
    required_columns: list of required column names
    min_rows: minimum number of rows required
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(dataframe, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(dataframe) < min_rows:
        return False, f"DataFrame must have at least {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in dataframe.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the Interquartile Range method.
    
    Parameters:
    data (numpy.ndarray): Input data array
    column (int): Index of column to clean
    
    Returns:
    numpy.ndarray: Data with outliers removed
    """
    if data.size == 0:
        return data
    
    col_data = data[:, column]
    q1 = np.percentile(col_data, 25)
    q3 = np.percentile(col_data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    mask = (col_data >= lower_bound) & (col_data <= upper_bound)
    return data[mask]

def calculate_statistics(data, column):
    """
    Calculate basic statistics for a column after outlier removal.
    
    Parameters:
    data (numpy.ndarray): Input data array
    column (int): Index of column to analyze
    
    Returns:
    dict: Dictionary containing statistical measures
    """
    if data.size == 0:
        return {}
    
    col_data = data[:, column]
    stats = {
        'mean': np.mean(col_data),
        'median': np.median(col_data),
        'std': np.std(col_data),
        'min': np.min(col_data),
        'max': np.max(col_data),
        'count': len(col_data)
    }
    return stats

def clean_dataset(data, columns_to_clean=None):
    """
    Clean multiple columns in a dataset using IQR method.
    
    Parameters:
    data (numpy.ndarray): Input data array
    columns_to_clean (list): List of column indices to clean
    
    Returns:
    numpy.ndarray: Cleaned dataset
    """
    if columns_to_clean is None:
        columns_to_clean = list(range(data.shape[1]))
    
    cleaned_data = data.copy()
    for column in columns_to_clean:
        if column < data.shape[1]:
            cleaned_data = remove_outliers_iqr(cleaned_data, column)
    
    return cleaned_data

if __name__ == "__main__":
    # Example usage
    sample_data = np.random.randn(100, 3) * 10 + 50
    sample_data[0, 0] = 200  # Add an outlier
    
    print("Original data shape:", sample_data.shape)
    print("First few rows of original data:")
    print(sample_data[:5])
    
    cleaned = clean_dataset(sample_data, [0, 1, 2])
    
    print("\nCleaned data shape:", cleaned.shape)
    print("First few rows of cleaned data:")
    print(cleaned[:5])
    
    stats = calculate_statistics(cleaned, 0)
    print("\nStatistics for column 0 after cleaning:")
    for key, value in stats.items():
        print(f"{key}: {value:.4f}")
def remove_duplicates(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
def filter_none_values(data):
    """
    Filter out None values from a list.
    
    Args:
        data: A list containing any values.
    
    Returns:
        A new list with all None values removed.
    """
    return [item for item in data if item is not None]
import pandas as pd
import hashlib

def remove_duplicates(input_file, output_file, key_columns=None):
    """
    Remove duplicate rows from a CSV file based on specified columns.
    If no columns are specified, use all columns for comparison.
    """
    try:
        df = pd.read_csv(input_file)
        
        if key_columns is None:
            key_columns = df.columns.tolist()
        
        initial_count = len(df)
        
        df['_hash'] = df[key_columns].apply(
            lambda row: hashlib.md5(pd.util.hash_pandas_object(row).values.tobytes()).hexdigest(),
            axis=1
        )
        
        df_clean = df.drop_duplicates(subset=['_hash'], keep='first')
        df_clean = df_clean.drop(columns=['_hash'])
        
        final_count = len(df_clean)
        duplicates_removed = initial_count - final_count
        
        df_clean.to_csv(output_file, index=False)
        
        print(f"Processed: {input_file}")
        print(f"Initial rows: {initial_count}")
        print(f"Final rows: {final_count}")
        print(f"Duplicates removed: {duplicates_removed}")
        print(f"Cleaned data saved to: {output_file}")
        
        return df_clean
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return None
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate dataframe structure and content.
    """
    if df is None or df.empty:
        print("Error: DataFrame is empty or None.")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return False
    
    return True

if __name__ == "__main__":
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    
    cleaned_data = remove_duplicates(input_csv, output_csv)
    
    if cleaned_data is not None:
        is_valid = validate_dataframe(cleaned_data)
        if is_valid:
            print("Data validation passed.")
        else:
            print("Data validation failed.")
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, columns):
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def remove_outliers_zscore(df, columns, threshold=3):
    df_clean = df.copy()
    for col in columns:
        z_scores = np.abs(stats.zscore(df_clean[col]))
        df_clean = df_clean[z_scores < threshold]
    return df_clean

def normalize_minmax(df, columns):
    df_normalized = df.copy()
    for col in columns:
        min_val = df_normalized[col].min()
        max_val = df_normalized[col].max()
        df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
    return df_normalized

def normalize_zscore(df, columns):
    df_normalized = df.copy()
    for col in columns:
        mean_val = df_normalized[col].mean()
        std_val = df_normalized[col].std()
        df_normalized[col] = (df_normalized[col] - mean_val) / std_val
    return df_normalized

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize_method='minmax'):
    if outlier_method == 'iqr':
        df = remove_outliers_iqr(df, numeric_columns)
    elif outlier_method == 'zscore':
        df = remove_outliers_zscore(df, numeric_columns)
    
    if normalize_method == 'minmax':
        df = normalize_minmax(df, numeric_columns)
    elif normalize_method == 'zscore':
        df = normalize_zscore(df, numeric_columns)
    
    return df