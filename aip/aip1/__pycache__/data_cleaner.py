import pandas as pd
import numpy as np

def clean_csv_data(file_path, missing_strategy='mean', columns_to_drop=None):
    """
    Load and clean CSV data by handling missing values and optionally dropping columns.
    
    Parameters:
    file_path (str): Path to the CSV file.
    missing_strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop').
    columns_to_drop (list): List of column names to drop from the dataset.
    
    Returns:
    pandas.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at path: {file_path}")
    except Exception as e:
        raise Exception(f"Error reading CSV file: {e}")

    if columns_to_drop:
        df = df.drop(columns=columns_to_drop, errors='ignore')

    numeric_columns = df.select_dtypes(include=[np.number]).columns
    categorical_columns = df.select_dtypes(exclude=[np.number]).columns

    if missing_strategy == 'mean':
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    elif missing_strategy == 'median':
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
    elif missing_strategy == 'mode':
        for col in numeric_columns:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)
    elif missing_strategy == 'drop':
        df = df.dropna(subset=numeric_columns)
    else:
        raise ValueError("Invalid missing_strategy. Choose from 'mean', 'median', 'mode', 'drop'.")

    for col in categorical_columns:
        df[col] = df[col].fillna('Unknown')

    return df

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers from a specified column using the Interquartile Range (IQR) method.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame.
    column (str): Column name to process.
    multiplier (float): IQR multiplier for outlier detection.
    
    Returns:
    pandas.DataFrame: DataFrame with outliers removed.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, np.nan, 4, 5, 100],
        'B': [10, 20, 30, np.nan, 50, 60],
        'C': ['X', 'Y', np.nan, 'Z', 'X', 'Y']
    }
    df = pd.DataFrame(sample_data)
    df.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_csv_data('sample_data.csv', missing_strategy='mean', columns_to_drop=None)
    print("Cleaned DataFrame:")
    print(cleaned_df)
    
    filtered_df = remove_outliers_iqr(cleaned_df, 'A')
    print("\nDataFrame after outlier removal in column 'A':")
    print(filtered_df)
import pandas as pd
import numpy as np

def remove_missing_rows(df, threshold=0.5):
    """
    Remove rows with missing values exceeding threshold percentage.
    
    Args:
        df: pandas DataFrame
        threshold: float between 0 and 1
    
    Returns:
        Cleaned DataFrame
    """
    missing_per_row = df.isnull().mean(axis=1)
    return df[missing_per_row <= threshold].copy()

def fill_missing_with_median(df, columns=None):
    """
    Fill missing values with column median.
    
    Args:
        df: pandas DataFrame
        columns: list of column names or None for all numeric columns
    
    Returns:
        DataFrame with filled values
    """
    df_filled = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df.columns:
            median_val = df[col].median()
            df_filled[col] = df[col].fillna(median_val)
    
    return df_filled

def remove_outliers_iqr(df, columns=None, multiplier=1.5):
    """
    Remove outliers using IQR method.
    
    Args:
        df: pandas DataFrame
        columns: list of column names or None for all numeric columns
        multiplier: IQR multiplier for outlier detection
    
    Returns:
        DataFrame without outliers
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
            df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def standardize_columns(df, columns=None):
    """
    Standardize numeric columns to have zero mean and unit variance.
    
    Args:
        df: pandas DataFrame
        columns: list of column names or None for all numeric columns
    
    Returns:
        DataFrame with standardized columns
    """
    df_standardized = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val > 0:
                df_standardized[col] = (df[col] - mean_val) / std_val
    
    return df_standardized

def clean_dataset(df, missing_threshold=0.3, outlier_multiplier=1.5):
    """
    Complete data cleaning pipeline.
    
    Args:
        df: pandas DataFrame
        missing_threshold: threshold for removing rows with missing values
        outlier_multiplier: IQR multiplier for outlier detection
    
    Returns:
        Cleaned DataFrame
    """
    print(f"Original shape: {df.shape}")
    
    # Step 1: Remove rows with too many missing values
    df_clean = remove_missing_rows(df, threshold=missing_threshold)
    print(f"After removing missing rows: {df_clean.shape}")
    
    # Step 2: Fill remaining missing values
    df_clean = fill_missing_with_median(df_clean)
    
    # Step 3: Remove outliers
    df_clean = remove_outliers_iqr(df_clean, multiplier=outlier_multiplier)
    print(f"After removing outliers: {df_clean.shape}")
    
    # Step 4: Standardize numeric columns
    df_clean = standardize_columns(df_clean)
    
    return df_cleanimport pandas as pd
import numpy as np

def clean_data(input_file, output_file):
    """
    Load a CSV file, clean the data by handling missing values,
    removing duplicates, and standardizing text columns.
    """
    try:
        df = pd.read_csv(input_file)
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        # Fill missing numeric values with column median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        
        # Fill missing categorical values with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        # Standardize text columns: strip whitespace and convert to lowercase
        for col in categorical_cols:
            df[col] = df[col].astype(str).str.strip().str.lower()
        
        # Remove rows where all values are NaN
        df = df.dropna(how='all')
        
        # Save cleaned data
        df.to_csv(output_file, index=False)
        print(f"Data cleaned successfully. Output saved to {output_file}")
        return df
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except pd.errors.EmptyDataError:
        print("Error: Input file is empty.")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    clean_data('raw_data.csv', 'cleaned_data.csv')
import pandas as pd
import numpy as np

def remove_missing_rows(df, columns=None):
    """
    Remove rows with missing values from DataFrame.
    If columns specified, only check those columns.
    """
    if columns:
        return df.dropna(subset=columns)
    return df.dropna()

def fill_missing_with_mean(df, columns):
    """
    Fill missing values in specified columns with column mean.
    """
    df_filled = df.copy()
    for col in columns:
        if col in df.columns:
            mean_val = df[col].mean()
            df_filled[col] = df[col].fillna(mean_val)
    return df_filled

def detect_outliers_iqr(df, column, threshold=1.5):
    """
    Detect outliers using IQR method.
    Returns boolean Series where True indicates outlier.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    return (df[column] < lower_bound) | (df[column] > upper_bound)

def cap_outliers(df, column, method='iqr', threshold=1.5):
    """
    Cap outliers to specified bounds.
    Supports IQR method for outlier detection.
    """
    df_capped = df.copy()
    
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        df_capped[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
        df_capped[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    
    return df_capped

def normalize_column(df, column, method='minmax'):
    """
    Normalize column values.
    Supports min-max and z-score normalization.
    """
    df_normalized = df.copy()
    
    if method == 'minmax':
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val != min_val:
            df_normalized[column] = (df[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean_val = df[column].mean()
        std_val = df[column].std()
        if std_val != 0:
            df_normalized[column] = (df[column] - mean_val) / std_val
    
    return df_normalized

def clean_dataset(df, missing_strategy='remove', outlier_strategy='cap'):
    """
    Comprehensive data cleaning pipeline.
    """
    cleaned_df = df.copy()
    
    if missing_strategy == 'remove':
        cleaned_df = remove_missing_rows(cleaned_df)
    elif missing_strategy == 'mean':
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        cleaned_df = fill_missing_with_mean(cleaned_df, numeric_cols)
    
    if outlier_strategy == 'cap':
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            cleaned_df = cap_outliers(cleaned_df, col)
    
    return cleaned_df