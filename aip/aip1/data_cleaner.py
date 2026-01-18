
def remove_duplicates(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return resultimport pandas as pd
import numpy as np

def clean_csv_data(filepath, fill_strategy='mean', drop_threshold=0.5):
    """
    Load and clean CSV data by handling missing values and removing columns
    with excessive missing data.
    
    Parameters:
    filepath (str): Path to the CSV file
    fill_strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', 'zero')
    drop_threshold (float): Threshold for dropping columns (0.0 to 1.0)
    
    Returns:
    pandas.DataFrame: Cleaned dataframe
    """
    try:
        df = pd.read_csv(filepath)
        
        # Calculate missing percentage per column
        missing_percent = df.isnull().sum() / len(df)
        
        # Drop columns with missing data above threshold
        columns_to_drop = missing_percent[missing_percent > drop_threshold].index
        df = df.drop(columns=columns_to_drop)
        
        # Fill remaining missing values based on strategy
        if fill_strategy == 'mean':
            df = df.fillna(df.mean(numeric_only=True))
        elif fill_strategy == 'median':
            df = df.fillna(df.median(numeric_only=True))
        elif fill_strategy == 'mode':
            df = df.fillna(df.mode().iloc[0])
        elif fill_strategy == 'zero':
            df = df.fillna(0)
        else:
            raise ValueError(f"Unsupported fill strategy: {fill_strategy}")
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        # Reset index after cleaning
        df = df.reset_index(drop=True)
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate dataframe structure and content.
    
    Parameters:
    df (pandas.DataFrame): Dataframe to validate
    required_columns (list): List of required column names
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    if df is None or df.empty:
        print("Error: Dataframe is empty or None")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return False
    
    # Check for infinite values
    if np.any(np.isinf(df.select_dtypes(include=[np.number]))):
        print("Warning: Dataframe contains infinite values")
    
    return True

def save_cleaned_data(df, output_path):
    """
    Save cleaned dataframe to CSV file.
    
    Parameters:
    df (pandas.DataFrame): Cleaned dataframe
    output_path (str): Path to save the cleaned data
    
    Returns:
    bool: True if save successful, False otherwise
    """
    try:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving cleaned data: {str(e)}")
        return False