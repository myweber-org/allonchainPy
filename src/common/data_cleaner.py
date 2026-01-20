
import pandas as pd

def clean_dataframe(df, subset=None, fill_method='ffill'):
    """
    Clean a pandas DataFrame by removing duplicate rows and filling missing values.

    Parameters:
    df (pd.DataFrame): The input DataFrame to clean.
    subset (list, optional): Column labels to consider for identifying duplicates.
                             If None, all columns are used.
    fill_method (str, optional): Method to fill missing values.
                                 Options: 'ffill' (forward fill), 'bfill' (backward fill),
                                 or a scalar value to fill with a constant.
                                 Default is 'ffill'.

    Returns:
    pd.DataFrame: The cleaned DataFrame.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates(subset=subset, keep='first')

    # Fill missing values
    if fill_method in ['ffill', 'bfill']:
        df_cleaned = df_cleaned.fillna(method=fill_method)
    else:
        df_cleaned = df_cleaned.fillna(fill_method)

    return df_cleaned

if __name__ == "__main__":
    # Example usage
    data = {
        'A': [1, 2, 2, 4, None],
        'B': [5, None, 7, 7, 9],
        'C': ['x', 'y', 'y', 'z', 'z']
    }
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame (forward fill):")
    cleaned_df = clean_dataframe(df, subset=['A', 'B'], fill_method='ffill')
    print(cleaned_df)