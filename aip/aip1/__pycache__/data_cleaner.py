
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column of a dataset using the IQR method.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in the dataset.")

    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data