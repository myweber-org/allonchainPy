
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, multiplier=1.5):
    """
    Remove outliers using IQR method
    """
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    return filtered_df

def z_score_normalize(dataframe, column):
    """
    Normalize data using z-score method
    """
    mean_val = dataframe[column].mean()
    std_val = dataframe[column].std()
    
    if std_val > 0:
        dataframe[column + '_normalized'] = (dataframe[column] - mean_val) / std_val
    else:
        dataframe[column + '_normalized'] = 0
    
    return dataframe

def min_max_normalize(dataframe, column, feature_range=(0, 1)):
    """
    Normalize data using min-max scaling
    """
    min_val = dataframe[column].min()
    max_val = dataframe[column].max()
    
    if max_val > min_val:
        dataframe[column + '_scaled'] = ((dataframe[column] - min_val) / 
                                        (max_val - min_val)) * (feature_range[1] - feature_range[0]) + feature_range[0]
    else:
        dataframe[column + '_scaled'] = feature_range[0]
    
    return dataframe

def handle_missing_values(dataframe, column, strategy='mean'):
    """
    Handle missing values with different strategies
    """
    if strategy == 'mean':
        fill_value = dataframe[column].mean()
    elif strategy == 'median':
        fill_value = dataframe[column].median()
    elif strategy == 'mode':
        fill_value = dataframe[column].mode()[0]
    elif strategy == 'constant':
        fill_value = 0
    else:
        fill_value = dataframe[column].mean()
    
    dataframe[column] = dataframe[column].fillna(fill_value)
    return dataframe

def create_data_summary(dataframe):
    """
    Create comprehensive data summary
    """
    summary = {
        'total_rows': len(dataframe),
        'total_columns': len(dataframe.columns),
        'missing_values': dataframe.isnull().sum().sum(),
        'duplicate_rows': dataframe.duplicated().sum(),
        'numeric_columns': list(dataframe.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(dataframe.select_dtypes(include=['object']).columns)
    }
    
    return summary

def validate_dataframe(dataframe, required_columns=None):
    """
    Validate dataframe structure and content
    """
    validation_result = {
        'is_valid': True,
        'errors': []
    }
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in dataframe.columns]
        if missing_columns:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"Missing required columns: {missing_columns}")
    
    if dataframe.empty:
        validation_result['is_valid'] = False
        validation_result['errors'].append("DataFrame is empty")
    
    return validation_result