
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def clean_dataset(df, numeric_columns):
    original_shape = df.shape
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    removed_count = original_shape[0] - cleaned_df.shape[0]
    print(f"Removed {removed_count} outliers from dataset")
    return cleaned_df
import pandas as pd
import numpy as np
from pathlib import Path

def clean_csv_data(input_path, output_path=None, fill_strategy='mean'):
    """
    Clean CSV data by handling missing values and removing duplicates.
    
    Args:
        input_path (str): Path to input CSV file
        output_path (str, optional): Path for cleaned output CSV. 
                                     If None, adds '_cleaned' suffix to input filename.
        fill_strategy (str): Strategy for filling missing values. 
                             Options: 'mean', 'median', 'mode', 'zero', 'drop'
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    df = pd.read_csv(input_path)
    
    original_rows = len(df)
    original_columns = len(df.columns)
    
    df = df.drop_duplicates()
    
    if fill_strategy == 'drop':
        df = df.dropna()
    else:
        for column in df.select_dtypes(include=[np.number]).columns:
            if df[column].isnull().any():
                if fill_strategy == 'mean':
                    fill_value = df[column].mean()
                elif fill_strategy == 'median':
                    fill_value = df[column].median()
                elif fill_strategy == 'mode':
                    fill_value = df[column].mode()[0]
                elif fill_strategy == 'zero':
                    fill_value = 0
                else:
                    raise ValueError(f"Unknown fill strategy: {fill_strategy}")
                
                df[column] = df[column].fillna(fill_value)
    
    if output_path is None:
        input_path_obj = Path(input_path)
        output_path = input_path_obj.parent / f"{input_path_obj.stem}_cleaned.csv"
    
    df.to_csv(output_path, index=False)
    
    print(f"Data cleaning completed:")
    print(f"  Original rows: {original_rows}, Cleaned rows: {len(df)}")
    print(f"  Removed duplicates: {original_rows - len(df)}")
    print(f"  Output saved to: {output_path}")
    
    return df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list, optional): List of required column names
    
    Returns:
        dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'issues': [],
        'summary': {}
    }
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Missing required columns: {missing_columns}")
    
    validation_results['summary']['total_rows'] = len(df)
    validation_results['summary']['total_columns'] = len(df.columns)
    validation_results['summary']['numeric_columns'] = len(df.select_dtypes(include=[np.number]).columns)
    validation_results['summary']['categorical_columns'] = len(df.select_dtypes(include=['object']).columns)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        validation_results['summary']['has_numeric'] = True
        validation_results['summary']['numeric_stats'] = df[numeric_cols].describe().to_dict()
    else:
        validation_results['summary']['has_numeric'] = False
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'value': [10.5, None, 15.2, 20.1, None, 10.5],
        'category': ['A', 'B', 'A', 'C', 'B', 'A']
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned_df = clean_csv_data('test_data.csv', fill_strategy='mean')
    
    validation = validate_dataframe(cleaned_df, required_columns=['id', 'value', 'category'])
    print("\nValidation Results:")
    print(f"Valid: {validation['is_valid']}")
    print(f"Issues: {validation['issues']}")
    print(f"Summary: {validation['summary']}")