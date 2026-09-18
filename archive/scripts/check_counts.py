import pandas as pd
import numpy as np
import os

def load_data(file_path):
    if file_path.endswith('.parquet'):
        return pd.read_parquet(file_path)
    return pd.read_csv(file_path)

df = load_data("yelpzip_sampled.parquet")
df['date'] = pd.to_datetime(df['date'])
df['ym'] = df['date'].dt.to_period('M')

if os.path.exists("burst_months.csv"):
    burst_months_df = pd.read_csv("burst_months.csv")
    burst_months = pd.PeriodIndex(burst_months_df.iloc[:, 0], freq='M')
    df['is_burst'] = df['ym'].isin(burst_months)
    
    burst_count = len(df[df['is_burst']])
    normal_count = len(df[~df['is_burst']])
    print(f"Sample - Burst reviews: {burst_count}")
    print(f"Sample - Normal reviews: {normal_count}")
else:
    print("burst_months.csv not found")
