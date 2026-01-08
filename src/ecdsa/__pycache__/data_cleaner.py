import pandas as pd
import numpy as np

def clean_csv_data(filepath, fill_strategy='mean', drop_threshold=0.5):
    """
    Load and clean CSV data by handling missing values and removing invalid columns.
    
    Args:
        filepath: Path to the CSV file
        fill_strategy: Strategy for filling missing values ('mean', 'median', 'mode', 'zero')
        drop_threshold: Threshold of missing values to drop a column (0.0 to 1.0)
    
    Returns:
        Cleaned DataFrame and cleaning report dictionary
    """
    try:
        df = pd.read_csv(filepath)
        original_shape = df.shape
        
        cleaning_report = {
            'original_rows': original_shape[0],
            'original_columns': original_shape[1],
            'missing_values_before': df.isnull().sum().sum(),
            'actions_taken': []
        }
        
        # Remove columns with too many missing values
        missing_percent = df.isnull().mean()
        columns_to_drop = missing_percent[missing_percent > drop_threshold].index.tolist()
        
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
            cleaning_report['actions_taken'].append(
                f"Dropped columns with >{drop_threshold*100:.0f}% missing values: {columns_to_drop}"
            )
        
        # Fill remaining missing values based on strategy
        if fill_strategy == 'mean':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif fill_strategy == 'median':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif fill_strategy == 'mode':
            for col in df.columns:
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col] = df[col].fillna(mode_val.iloc[0])
        elif fill_strategy == 'zero':
            df = df.fillna(0)
        
        if df.isnull().sum().sum() > 0:
            df = df.dropna()
            cleaning_report['actions_taken'].append("Dropped rows with remaining missing values")
        
        cleaning_report.update({
            'final_rows': df.shape[0],
            'final_columns': df.shape[1],
            'missing_values_after': df.isnull().sum().sum(),
            'rows_removed': original_shape[0] - df.shape[0],
            'columns_removed': original_shape[1] - df.shape[1]
        })
        
        return df, cleaning_report
        
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except pd.errors.EmptyDataError:
        raise ValueError("The CSV file is empty")
    except Exception as e:
        raise RuntimeError(f"Error cleaning data: {str(e)}")

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
    
    Returns:
        Tuple of (is_valid, error_message)
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

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, np.nan, np.nan, 10, 11],
        'C': [7, 8, 9, np.nan, 11],
        'D': [12, 13, 14, 15, 16]
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned_df, report = clean_csv_data('test_data.csv', fill_strategy='mean', drop_threshold=0.6)
    
    print("Cleaning Report:")
    for key, value in report.items():
        print(f"{key}: {value}")
    
    is_valid, message = validate_dataframe(cleaned_df, min_rows=3)
    print(f"\nValidation: {message}")
    
    import os
    os.remove('test_data.csv')