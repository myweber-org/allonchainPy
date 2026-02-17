
import pandas as pd

def clean_dataframe(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.

    Parameters:
    df (pd.DataFrame): The input DataFrame to clean.
    drop_duplicates (bool): If True, drop duplicate rows.
    fill_missing (str): Method to fill missing values.
                        Options: 'mean', 'median', 'mode', or 'drop'.

    Returns:
    pd.DataFrame: The cleaned DataFrame.
    """
    cleaned_df = df.copy()

    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()

    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
            elif fill_missing == 'median':
                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col].fillna(cleaned_df[col].mode()[0], inplace=True)

    return cleaned_df

def main():
    sample_data = {
        'A': [1, 2, 2, None, 5],
        'B': [10, None, 10, 40, 50],
        'C': ['x', 'y', 'x', 'y', None]
    }
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)

    cleaned = clean_dataframe(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)

if __name__ == "__main__":
    main()