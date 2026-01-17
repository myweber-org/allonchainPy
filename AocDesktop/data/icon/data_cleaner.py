
import pandas as pd
import numpy as np

def clean_data(df, missing_strategy='mean', outlier_method='iqr'):
    """
    Clean a DataFrame by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    missing_strategy (str): Strategy for missing values ('mean', 'median', 'mode', 'drop').
    outlier_method (str): Method for outlier detection ('iqr', 'zscore').
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if missing_strategy == 'mean':
        cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
    elif missing_strategy == 'median':
        cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
    elif missing_strategy == 'mode':
        cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
    elif missing_strategy == 'drop':
        cleaned_df = cleaned_df.dropna()
    
    # Handle outliers for numeric columns
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    
    if outlier_method == 'iqr':
        for col in numeric_cols:
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            cleaned_df[col] = np.where((cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound),
                                       cleaned_df[col].median(), cleaned_df[col])
    elif outlier_method == 'zscore':
        for col in numeric_cols:
            z_scores = np.abs((cleaned_df[col] - cleaned_df[col].mean()) / cleaned_df[col].std())
            cleaned_df[col] = np.where(z_scores > 3, cleaned_df[col].median(), cleaned_df[col])
    
    return cleaned_df

def main():
    # Example usage
    data = {
        'A': [1, 2, np.nan, 4, 100],
        'B': [5, 6, 7, np.nan, 9],
        'C': [10, 11, 12, 13, 14]
    }
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_data(df, missing_strategy='mean', outlier_method='iqr')
    print("\nCleaned DataFrame:")
    print(cleaned)

if __name__ == "__main__":
    main()