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
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(input_file, output_file):
    df = pd.read_csv(input_file)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        df = remove_outliers_iqr(df, col)
    
    for col in numeric_columns:
        df = normalize_minmax(df, col)
    
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")
    print(f"Original shape: {pd.read_csv(input_file).shape}")
    print(f"Cleaned shape: {df.shape}")

if __name__ == "__main__":
    clean_dataset('raw_data.csv', 'cleaned_data.csv')
import pandas as pd
import numpy as np

def clean_dataset(df, columns_to_check=None, fill_missing=True):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    original_shape = df.shape
    
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()
    duplicates_removed = original_shape[0] - df_cleaned.shape[0]
    
    # Handle missing values
    if fill_missing:
        if columns_to_check is None:
            columns_to_check = df_cleaned.columns
        
        for column in columns_to_check:
            if df_cleaned[column].dtype in [np.float64, np.int64]:
                # Fill numeric columns with median
                df_cleaned[column] = df_cleaned[column].fillna(df_cleaned[column].median())
            elif df_cleaned[column].dtype == 'object':
                # Fill categorical columns with mode
                df_cleaned[column] = df_cleaned[column].fillna(df_cleaned[column].mode()[0] if not df_cleaned[column].mode().empty else 'Unknown')
    
    # Reset index after cleaning
    df_cleaned = df_cleaned.reset_index(drop=True)
    
    # Print cleaning summary
    print(f"Original dataset shape: {original_shape}")
    print(f"Cleaned dataset shape: {df_cleaned.shape}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Missing values handled: {fill_missing}")
    
    return df_cleaned

def validate_data(df, required_columns=None):
    """
    Validate that the DataFrame meets certain criteria.
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for any remaining NaN values
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        print(f"Warning: Dataset still contains {nan_count} NaN values")
    
    return True

# Example usage function
def process_data_file(file_path, output_path=None):
    """
    Load, clean, and save a dataset from a CSV file.
    """
    try:
        # Load data
        df = pd.read_csv(file_path)
        
        # Clean data
        df_cleaned = clean_dataset(df)
        
        # Validate cleaned data
        validate_data(df_cleaned)
        
        # Save cleaned data if output path provided
        if output_path:
            df_cleaned.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
        
        return df_cleaned
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None