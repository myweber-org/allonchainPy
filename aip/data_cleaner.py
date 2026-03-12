
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