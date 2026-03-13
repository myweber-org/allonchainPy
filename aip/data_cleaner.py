
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
    if max_val != min_val:
        df[column] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df = normalize_minmax(cleaned_df, col)
    return cleaned_df.reset_index(drop=True)

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 200),
        'feature_b': np.random.exponential(50, 200),
        'category': np.random.choice(['X', 'Y', 'Z'], 200)
    })
    cleaned = clean_dataset(sample_data, ['feature_a', 'feature_b'])
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {cleaned.shape}")
    print(cleaned.head())
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column in a dataset using the IQR method.
    
    Parameters:
    data (pandas.DataFrame): The input dataset.
    column (str): The column name to process.
    
    Returns:
    pandas.DataFrame: Dataset with outliers removed from the specified column.
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
from scipy import stats

def load_dataset(filepath):
    """Load dataset from CSV file."""
    return pd.read_csv(filepath)

def remove_outliers_iqr(df, columns):
    """Remove outliers using IQR method."""
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def normalize_data(df, columns, method='minmax'):
    """Normalize specified columns."""
    df_norm = df.copy()
    for col in columns:
        if method == 'minmax':
            df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())
        elif method == 'zscore':
            df_norm[col] = stats.zscore(df_norm[col])
        else:
            raise ValueError("Method must be 'minmax' or 'zscore'")
    return df_norm

def handle_missing_values(df, strategy='mean'):
    """Handle missing values in numeric columns."""
    df_filled = df.copy()
    numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if strategy == 'mean':
            fill_value = df_filled[col].mean()
        elif strategy == 'median':
            fill_value = df_filled[col].median()
        elif strategy == 'mode':
            fill_value = df_filled[col].mode()[0]
        else:
            raise ValueError("Strategy must be 'mean', 'median', or 'mode'")
        
        df_filled[col] = df_filled[col].fillna(fill_value)
    
    return df_filled

def clean_dataset(input_path, output_path, numeric_columns):
    """Complete data cleaning pipeline."""
    df = load_dataset(input_path)
    df = handle_missing_values(df, strategy='median')
    df = remove_outliers_iqr(df, numeric_columns)
    df = normalize_data(df, numeric_columns, method='minmax')
    df.to_csv(output_path, index=False)
    return df

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 200),
        'feature2': np.random.exponential(50, 200),
        'feature3': np.random.uniform(0, 1, 200)
    })
    
    sample_data.iloc[10:15, 0] = np.nan
    sample_data.iloc[5, 1] = 500
    
    sample_data.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_dataset(
        'sample_data.csv',
        'cleaned_data.csv',
        ['feature1', 'feature2', 'feature3']
    )
    
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")
    print("Data cleaning completed successfully.")
def remove_duplicates(seq):
    seen = set()
    result = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return resultimport re
import pandas as pd

def remove_special_characters(text):
    """Remove non-alphanumeric characters from a string."""
    if pd.isna(text):
        return text
    return re.sub(r'[^A-Za-z0-9\s]+', '', str(text))

def validate_email(email):
    """Validate an email address format."""
    if pd.isna(email):
        return False
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, str(email)))

def clean_dataframe(df, columns_to_clean=None):
    """Clean a DataFrame by removing special characters from specified columns."""
    df_clean = df.copy()
    if columns_to_clean is None:
        columns_to_clean = df_clean.select_dtypes(include=['object']).columns
    
    for col in columns_to_clean:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(remove_special_characters)
    
    return df_clean

def add_valid_email_column(df, email_column):
    """Add a boolean column indicating if the email in the specified column is valid."""
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    df_copy = df.copy()
    new_column_name = f"{email_column}_valid"
    df_copy[new_column_name] = df_copy[email_column].apply(validate_email)
    return df_copy
import pandas as pd

def clean_dataset(df, remove_duplicates=True, fill_method='drop'):
    """
    Clean a pandas DataFrame by handling missing values and optionally removing duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    remove_duplicates (bool): If True, remove duplicate rows.
    fill_method (str): Method to handle missing values: 'drop', 'fill_mean', 'fill_median', 'fill_mode'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if fill_method == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_method == 'fill_mean':
        cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
    elif fill_method == 'fill_median':
        cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
    elif fill_method == 'fill_mode':
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
    else:
        raise ValueError("Invalid fill_method. Choose from 'drop', 'fill_mean', 'fill_median', 'fill_mode'.")
    
    if remove_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and required columns.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    bool: True if validation passes, raises ValueError otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, None, 4, 2],
        'B': [5, None, 7, 8, 5],
        'C': ['x', 'y', 'z', None, 'x']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, remove_duplicates=True, fill_method='fill_mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
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
        'count': df[column].count()
    }
    
    return stats

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    If no columns specified, clean all numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    cleaned_df = df.copy()
    
    for column in columns:
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            try:
                cleaned_df = remove_outliers_iqr(cleaned_df, column)
            except Exception as e:
                print(f"Warning: Could not process column '{column}': {e}")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.randint(1, 100, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    df.loc[0:10, 'A'] = 500
    
    print("Original DataFrame shape:", df.shape)
    print("\nOriginal statistics for column 'A':")
    print(calculate_basic_stats(df, 'A'))
    
    cleaned_df = clean_numeric_data(df, ['A', 'B'])
    
    print("\nCleaned DataFrame shape:", cleaned_df.shape)
    print("\nCleaned statistics for column 'A':")
    print(calculate_basic_stats(cleaned_df, 'A'))