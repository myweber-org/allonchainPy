
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_outliers_iqr(self, columns=None, factor=1.5):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        clean_df = self.df.copy()
        for col in columns:
            if col in clean_df.columns:
                Q1 = clean_df[col].quantile(0.25)
                Q3 = clean_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                clean_df = clean_df[(clean_df[col] >= lower_bound) & (clean_df[col] <= upper_bound)]
        
        removed_count = len(self.df) - len(clean_df)
        self.df = clean_df
        return removed_count
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        normalized_df = self.df.copy()
        for col in columns:
            if col in normalized_df.columns:
                min_val = normalized_df[col].min()
                max_val = normalized_df[col].max()
                if max_val > min_val:
                    normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
        
        self.df = normalized_df
        return self.df
    
    def standardize_zscore(self, columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        standardized_df = self.df.copy()
        for col in columns:
            if col in standardized_df.columns:
                mean_val = standardized_df[col].mean()
                std_val = standardized_df[col].std()
                if std_val > 0:
                    standardized_df[col] = (standardized_df[col] - mean_val) / std_val
        
        self.df = standardized_df
        return self.df
    
    def handle_missing_values(self, strategy='mean', columns=None):
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        filled_df = self.df.copy()
        for col in columns:
            if col in filled_df.columns and filled_df[col].isnull().any():
                if strategy == 'mean':
                    fill_value = filled_df[col].mean()
                elif strategy == 'median':
                    fill_value = filled_df[col].median()
                elif strategy == 'mode':
                    fill_value = filled_df[col].mode()[0]
                else:
                    fill_value = 0
                
                filled_df[col] = filled_df[col].fillna(fill_value)
        
        self.df = filled_df
        return self.df
    
    def get_cleaned_data(self):
        return self.df
    
    def get_removed_percentage(self):
        original_rows = self.original_shape[0]
        current_rows = len(self.df)
        return ((original_rows - current_rows) / original_rows) * 100

def create_sample_data():
    np.random.seed(42)
    data = {
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.uniform(0, 200, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    }
    
    df = pd.DataFrame(data)
    df.loc[np.random.choice(df.index, 50), 'feature_a'] = np.nan
    df.loc[np.random.choice(df.index, 30), 'feature_b'] = np.nan
    
    outliers = np.random.randint(0, 1000, 20)
    df.loc[outliers, 'feature_c'] = df['feature_c'].max() * 5
    
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    cleaner = DataCleaner(sample_df)
    
    print(f"Original shape: {cleaner.original_shape}")
    print(f"Missing values before: {sample_df.isnull().sum().sum()}")
    
    cleaner.handle_missing_values(strategy='mean')
    removed = cleaner.remove_outliers_iqr(factor=1.5)
    cleaner.normalize_minmax()
    
    cleaned_df = cleaner.get_cleaned_data()
    print(f"Removed outliers: {removed}")
    print(f"Cleaned shape: {cleaned_df.shape}")
    print(f"Removed percentage: {cleaner.get_removed_percentage():.2f}%")
    print(f"Missing values after: {cleaned_df.isnull().sum().sum()}")
    print(f"Data range after normalization: [{cleaned_df['feature_a'].min():.4f}, {cleaned_df['feature_a'].max():.4f}]")import pandas as pd
import numpy as np

def clean_data(input_file, output_file):
    """
    Load a CSV file, perform basic cleaning operations,
    and save the cleaned data to a new file.
    """
    try:
        df = pd.read_csv(input_file)
        print(f"Original data shape: {df.shape}")

        # Remove duplicate rows
        df = df.drop_duplicates()
        print(f"After removing duplicates: {df.shape}")

        # Fill missing numeric values with column median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"Filled missing values in {col} with median: {median_val}")

        # Remove rows where critical text columns are missing
        text_cols = df.select_dtypes(include=['object']).columns
        critical_text_cols = [col for col in text_cols if 'name' in col.lower() or 'id' in col.lower()]
        if critical_text_cols:
            df = df.dropna(subset=critical_text_cols)
            print(f"After removing rows with missing critical text: {df.shape}")

        # Reset index after cleaning
        df = df.reset_index(drop=True)

        # Save cleaned data
        df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to: {output_file}")
        return df

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print("Error: Input file is empty.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    cleaned_df = clean_data(input_csv, output_csv)
    if cleaned_df is not None:
        print("Data cleaning completed successfully.")