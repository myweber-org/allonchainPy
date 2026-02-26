
import pandas as pd
import numpy as np
import sys

def clean_csv(input_file, output_file):
    try:
        df = pd.read_csv(input_file)
        
        print(f"Original shape: {df.shape}")
        print(f"Missing values per column:")
        print(df.isnull().sum())
        
        df_cleaned = df.copy()
        
        for column in df_cleaned.columns:
            if df_cleaned[column].dtype in ['int64', 'float64']:
                df_cleaned[column].fillna(df_cleaned[column].median(), inplace=True)
            elif df_cleaned[column].dtype == 'object':
                df_cleaned[column].fillna(df_cleaned[column].mode()[0] if not df_cleaned[column].mode().empty else 'Unknown', inplace=True)
        
        df_cleaned = df_cleaned.drop_duplicates()
        
        numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
        if not numeric_columns.empty:
            Q1 = df_cleaned[numeric_columns].quantile(0.25)
            Q3 = df_cleaned[numeric_columns].quantile(0.75)
            IQR = Q3 - Q1
            
            outlier_condition = (
                (df_cleaned[numeric_columns] < (Q1 - 1.5 * IQR)) | 
                (df_cleaned[numeric_columns] > (Q3 + 1.5 * IQR))
            )
            
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].mask(
                outlier_condition, 
                df_cleaned[numeric_columns].median()
            )
        
        df_cleaned.to_csv(output_file, index=False)
        
        print(f"Cleaned shape: {df_cleaned.shape}")
        print(f"Cleaned data saved to: {output_file}")
        
        return True
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return False
    except Exception as e:
        print(f"Error during cleaning: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python data_cleaner.py <input_file.csv> <output_file.csv>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    success = clean_csv(input_file, output_file)
    
    if success:
        print("Data cleaning completed successfully.")
    else:
        print("Data cleaning failed.")
        sys.exit(1)
import pandas as pd
import numpy as np

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
    
    return filtered_df.reset_index(drop=True)

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

def clean_dataset(df, columns_to_clean=None):
    """
    Clean multiple columns in a DataFrame by removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns_to_clean (list): List of column names to clean. If None, clean all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    dict: Dictionary of removed outliers count per column
    """
    if columns_to_clean is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns_to_clean = list(numeric_cols)
    
    cleaned_df = df.copy()
    removed_stats = {}
    
    for column in columns_to_clean:
        if column in cleaned_df.columns:
            initial_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = initial_count - len(cleaned_df)
            removed_stats[column] = removed_count
    
    return cleaned_df, removed_stats