import pandas as pd

def clean_dataframe(df):
    """
    Remove duplicate rows and fill missing values with column mean for numeric columns,
    or mode for categorical columns.
    """
    # Remove duplicates
    df_cleaned = df.drop_duplicates()

    # Handle missing values
    for column in df_cleaned.columns:
        if df_cleaned[column].dtype in ['int64', 'float64']:
            # Numeric columns: fill with mean
            df_cleaned[column].fillna(df_cleaned[column].mean(), inplace=True)
        else:
            # Categorical columns: fill with mode
            if not df_cleaned[column].mode().empty:
                df_cleaned[column].fillna(df_cleaned[column].mode()[0], inplace=True)
            else:
                df_cleaned[column].fillna('Unknown', inplace=True)

    return df_cleaned

def validate_dataframe(df):
    """
    Validate that the dataframe has no missing values after cleaning.
    """
    missing_values = df.isnull().sum().sum()
    if missing_values == 0:
        print("Data validation passed: No missing values found.")
        return True
    else:
        print(f"Data validation failed: {missing_values} missing values found.")
        return False

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, 2, None, 5],
        'B': [None, 'x', 'y', 'y', 'z'],
        'C': [10.5, None, 30.2, 30.2, 50.0]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_dataframe(df)
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    validate_dataframe(cleaned_df)