
import numpy as np
import pandas as pd

def remove_outliers_iqr(dataframe, column):
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return dataframe[(dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)]

def normalize_column(dataframe, column):
    min_val = dataframe[column].min()
    max_val = dataframe[column].max()
    if max_val == min_val:
        return dataframe[column].apply(lambda x: 0.0)
    return (dataframe[column] - min_val) / (max_val - min_val)

def clean_dataset(dataframe, numeric_columns):
    cleaned_df = dataframe.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col] = normalize_column(cleaned_df, col)
    return cleaned_df

def generate_summary(dataframe):
    summary = {
        'original_rows': len(dataframe),
        'cleaned_rows': len(dataframe.dropna()),
        'numeric_columns': list(dataframe.select_dtypes(include=[np.number]).columns),
        'missing_values': dataframe.isnull().sum().to_dict()
    }
    return summary

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': np.random.normal(100, 15, 200),
        'B': np.random.exponential(2, 200),
        'C': np.random.randint(1, 50, 200)
    })
    
    sample_data.loc[10:15, 'A'] = 500
    sample_data.loc[20:25, 'B'] = 100
    
    print("Original dataset shape:", sample_data.shape)
    cleaned = clean_dataset(sample_data, ['A', 'B', 'C'])
    print("Cleaned dataset shape:", cleaned.shape)
    
    summary = generate_summary(cleaned)
    print("Dataset summary:", summary)