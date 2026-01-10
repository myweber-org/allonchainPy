
import pandas as pd
import numpy as np
from datetime import datetime

def clean_dataset(input_file, output_file):
    df = pd.read_csv(input_file)
    
    df.drop_duplicates(inplace=True)
    
    df.fillna({'price': 0, 'quantity': 1}, inplace=True)
    
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    df['category'] = df['category'].str.upper().str.strip()
    
    df['total'] = df['price'] * df['quantity']
    
    df = df[df['price'] >= 0]
    
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")
    print(f"Original rows: {len(df)}, Cleaned rows: {len(df)}")
    
    return df

if __name__ == "__main__":
    cleaned_df = clean_dataset('raw_data.csv', 'cleaned_data.csv')