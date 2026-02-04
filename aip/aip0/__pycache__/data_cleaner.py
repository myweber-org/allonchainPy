
import pandas as pd
import numpy as np

def clean_dataset(df):
    """
    Remove duplicate rows and standardize column names.
    """
    # Remove duplicates
    df_cleaned = df.drop_duplicates()
    
    # Standardize column names
    df_cleaned.columns = df_cleaned.columns.str.strip().str.lower().str.replace(' ', '_')
    
    return df_cleaned

def handle_missing_values(df, strategy='mean'):
    """
    Handle missing values using specified strategy.
    """
    if strategy == 'mean':
        return df.fillna(df.mean())
    elif strategy == 'median':
        return df.fillna(df.median())
    elif strategy == 'mode':
        return df.fillna(df.mode().iloc[0])
    else:
        return df.fillna(0)

def remove_outliers(df, column, threshold=3):
    """
    Remove outliers using z-score method.
    """
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    return df[z_scores < threshold]

def main():
    # Example usage
    data = pd.DataFrame({
        'ID': [1, 2, 3, 4, 5, 5],
        'Value': [10, 20, np.nan, 40, 50, 50],
        'Category': ['A', 'B', 'C', 'D', 'E', 'E']
    })
    
    print("Original Data:")
    print(data)
    
    cleaned_data = clean_dataset(data)
    print("\nAfter cleaning:")
    print(cleaned_data)
    
    filled_data = handle_missing_values(cleaned_data)
    print("\nAfter handling missing values:")
    print(filled_data)

if __name__ == "__main__":
    main()
import pandas as pd

def remove_duplicates(dataframe, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        dataframe: pandas DataFrame
        subset: column label or sequence of labels to consider for duplicates
        keep: determines which duplicates to keep ('first', 'last', False)
    
    Returns:
        Cleaned DataFrame with duplicates removed
    """
    if dataframe.empty:
        return dataframe
    
    cleaned_df = dataframe.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(dataframe) - len(cleaned_df)
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate row(s)")
    
    return cleaned_df

def validate_dataframe(dataframe, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        dataframe: pandas DataFrame to validate
        required_columns: list of column names that must be present
    
    Returns:
        Boolean indicating if DataFrame is valid
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if dataframe.empty:
        print("Warning: DataFrame is empty")
        return True
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in dataframe.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True

def clean_numeric_columns(dataframe, columns=None):
    """
    Clean numeric columns by converting to appropriate types.
    
    Args:
        dataframe: pandas DataFrame
        columns: list of column names to clean (defaults to all numeric columns)
    
    Returns:
        DataFrame with cleaned numeric columns
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=['number']).columns
    
    cleaned_df = dataframe.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
    
    return cleaned_df

def get_data_summary(dataframe):
    """
    Generate summary statistics for a DataFrame.
    
    Args:
        dataframe: pandas DataFrame
    
    Returns:
        Dictionary containing summary statistics
    """
    summary = {
        'rows': len(dataframe),
        'columns': len(dataframe.columns),
        'missing_values': dataframe.isnull().sum().sum(),
        'duplicates': dataframe.duplicated().sum(),
        'column_types': dataframe.dtypes.to_dict()
    }
    
    return summary
def remove_duplicates(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]