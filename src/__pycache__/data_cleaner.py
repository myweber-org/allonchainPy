
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

def calculate_summary_statistics(df, column):
    mean_val = df[column].mean()
    median_val = df[column].median()
    std_val = df[column].std()
    return {
        'mean': mean_val,
        'median': median_val,
        'std': std_val
    }

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'values': np.random.normal(100, 15, 1000)
    })
    cleaned_data = remove_outliers_iqr(sample_data, 'values')
    stats = calculate_summary_statistics(cleaned_data, 'values')
    print(f"Original data points: {len(sample_data)}")
    print(f"Cleaned data points: {len(cleaned_data)}")
    print(f"Summary statistics: {stats}")