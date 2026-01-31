import pandas as pd
import numpy as np

def load_data(filepath):
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

def remove_outliers(df, column, threshold=3):
    """Remove outliers using the Z-score method."""
    if column not in df.columns:
        print(f"Column '{column}' not found in DataFrame.")
        return df
    
    mean = df[column].mean()
    std = df[column].std()
    z_scores = np.abs((df[column] - mean) / std)
    filtered_df = df[z_scores < threshold]
    removed_count = len(df) - len(filtered_df)
    print(f"Removed {removed_count} outliers from column '{column}'.")
    return filtered_df

def normalize_column(df, column):
    """Normalize a column using Min-Max scaling."""
    if column not in df.columns:
        print(f"Column '{column}' not found in DataFrame.")
        return df
    
    min_val = df[column].min()
    max_val = df[column].max()
    if max_val == min_val:
        print(f"Column '{column}' has constant values. Normalization skipped.")
        return df
    
    df[column] = (df[column] - min_val) / (max_val - min_val)
    print(f"Column '{column}' normalized.")
    return df

def clean_data(df, numeric_columns):
    """Apply outlier removal and normalization to numeric columns."""
    if df is None or df.empty:
        print("DataFrame is empty or None.")
        return df
    
    for col in numeric_columns:
        if col in df.columns:
            df = remove_outliers(df, col)
            df = normalize_column(df, col)
        else:
            print(f"Column '{col}' not found. Skipping.")
    
    return df

def save_cleaned_data(df, output_path):
    """Save the cleaned DataFrame to a CSV file."""
    try:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")
    except Exception as e:
        print(f"Error saving data: {e}")

def main():
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    numeric_cols = ["age", "income", "score"]
    
    df = load_data(input_file)
    if df is not None:
        cleaned_df = clean_data(df, numeric_cols)
        save_cleaned_data(cleaned_df, output_file)

if __name__ == "__main__":
    main()