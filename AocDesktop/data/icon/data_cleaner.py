
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
    
    return filtered_df.reset_index(drop=True)

def calculate_statistics(df, column):
    """
    Calculate basic statistics for a column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
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
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

def clean_dataset(df, numeric_columns=None):
    """
    Clean dataset by removing outliers from all numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of numeric column names to clean
    
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
                print(f"Warning: Could not process column '{column}': {e}")
    
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
    sample_data.loc[95:99, 'value'] = [500, -200, 300, 1000, -500]
    
    print("Original dataset shape:", sample_data.shape)
    print("Original statistics:", calculate_statistics(sample_data, 'value'))
    
    cleaned_data = clean_dataset(sample_data, ['value'])
    
    print("\nCleaned dataset shape:", cleaned_data.shape)
    print("Cleaned statistics:", calculate_statistics(cleaned_data, 'value'))
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
    dict: Dictionary containing summary statistics
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        return {}
    
    stats = {}
    for col in numeric_cols:
        stats[col] = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'count': df[col].count()
        }
    
    return stats

def normalize_column(df, column):
    """
    Normalize a column using min-max scaling.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to normalize
    
    Returns:
    pd.DataFrame: DataFrame with normalized column
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' must be numeric")
    
    df_copy = df.copy()
    min_val = df_copy[column].min()
    max_val = df_copy[column].max()
    
    if max_val == min_val:
        df_copy[f'{column}_normalized'] = 0.5
    else:
        df_copy[f'{column}_normalized'] = (df_copy[column] - min_val) / (max_val - min_val)
    
    return df_copy

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    tuple: (bool, str) Validation result and message
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame validation passed"
def remove_duplicates(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]
import pandas as pd
import numpy as np
from datetime import datetime

def clean_dataframe(df, date_column=None, numeric_columns=None, text_columns=None):
    """
    Clean a pandas DataFrame by handling missing values, duplicates,
    and standardizing data formats.
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Remove duplicate rows
    initial_rows = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates()
    duplicates_removed = initial_rows - len(cleaned_df)
    
    # Standardize column names
    cleaned_df.columns = cleaned_df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # Handle missing values
    if numeric_columns:
        for col in numeric_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    
    if text_columns:
        for col in text_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].fillna('unknown').str.strip()
    
    # Standardize date formats
    if date_column and date_column in cleaned_df.columns:
        try:
            cleaned_df[date_column] = pd.to_datetime(cleaned_df[date_column], errors='coerce')
        except:
            print(f"Warning: Could not convert {date_column} to datetime")
    
    # Remove outliers using IQR method for numeric columns
    if numeric_columns:
        for col in numeric_columns:
            if col in cleaned_df.columns:
                Q1 = cleaned_df[col].quantile(0.25)
                Q3 = cleaned_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers instead of removing rows
                cleaned_df[col] = np.where(cleaned_df[col] < lower_bound, lower_bound, cleaned_df[col])
                cleaned_df[col] = np.where(cleaned_df[col] > upper_bound, upper_bound, cleaned_df[col])
    
    # Generate cleaning report
    report = {
        'original_rows': len(df),
        'cleaned_rows': len(cleaned_df),
        'duplicates_removed': duplicates_removed,
        'missing_values_filled': df.isnull().sum().sum() - cleaned_df.isnull().sum().sum(),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return cleaned_df, report

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Check if DataFrame is empty
    if len(df) < min_rows:
        validation_results['is_valid'] = False
        validation_results['errors'].append(f'DataFrame has fewer than {min_rows} rows')
    
    # Check required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f'Missing required columns: {missing_columns}')
    
    # Check for excessive missing values
    missing_percentage = df.isnull().sum() / len(df)
    high_missing = missing_percentage[missing_percentage > 0.5].index.tolist()
    if high_missing:
        validation_results['warnings'].append(f'Columns with >50% missing values: {high_missing}')
    
    return validation_results

# Example usage function
def process_data_file(file_path, output_path=None):
    """
    Complete data processing pipeline for a CSV file.
    """
    try:
        # Read data
        df = pd.read_csv(file_path)
        
        # Validate data
        validation = validate_dataframe(df)
        if not validation['is_valid']:
            print("Data validation failed:")
            for error in validation['errors']:
                print(f"  - {error}")
            return None
        
        # Clean data
        cleaned_df, report = clean_dataframe(
            df,
            date_column='date',
            numeric_columns=['price', 'quantity', 'rating'],
            text_columns=['product_name', 'category', 'supplier']
        )
        
        # Print cleaning report
        print("Data Cleaning Report:")
        print(f"  Original rows: {report['original_rows']}")
        print(f"  Cleaned rows: {report['cleaned_rows']}")
        print(f"  Duplicates removed: {report['duplicates_removed']}")
        print(f"  Missing values filled: {report['missing_values_filled']}")
        print(f"  Cleaning timestamp: {report['timestamp']}")
        
        # Save cleaned data
        if output_path:
            cleaned_df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
        
        return cleaned_df
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None

if __name__ == "__main__":
    # This would be used when running the script directly
    import sys
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = 'cleaned_data.csv' if len(sys.argv) < 3 else sys.argv[2]
        process_data_file(input_file, output_file)
    else:
        print("Usage: python data_cleaner.py <input_file> [output_file]")
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if min_val == max_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(data, numeric_columns=None, outlier_method='iqr', normalize_method='minmax'):
    """
    Clean dataset by removing outliers and normalizing numeric columns
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column not in data.columns:
            continue
            
        if outlier_method == 'iqr':
            cleaned_data = remove_outliers_iqr(cleaned_data, column)
        elif outlier_method == 'zscore':
            cleaned_data = remove_outliers_zscore(cleaned_data, column)
        
        if normalize_method == 'minmax':
            cleaned_data[f'{column}_normalized'] = normalize_minmax(cleaned_data, column)
        elif normalize_method == 'zscore':
            cleaned_data[f'{column}_standardized'] = normalize_zscore(cleaned_data, column)
    
    return cleaned_data

def validate_data(data, required_columns=None, check_missing=True, check_duplicates=True):
    """
    Validate dataset for common data quality issues
    """
    validation_report = {}
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            validation_report['missing_columns'] = missing_columns
    
    if check_missing:
        missing_values = data.isnull().sum()
        missing_percentage = (missing_values / len(data)) * 100
        validation_report['missing_values'] = missing_percentage[missing_percentage > 0].to_dict()
    
    if check_duplicates:
        duplicate_count = data.duplicated().sum()
        validation_report['duplicate_rows'] = duplicate_count
    
    return validation_report