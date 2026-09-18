import pandas as pd
import numpy as np

df = pd.read_parquet('yelpzip_sampled.parquet')
df['date'] = pd.to_datetime(df['date'])

fraud_user_set = set(df.loc[df['label'] == 1, 'user_id'])
df_fuser = df[df['user_id'].isin(fraud_user_set)].copy()

print(f'Total: {len(df):,}')
print(f'Fraud Labels: {len(df[df.label == 1]):,}')
print(f'Fraud Users: {len(fraud_user_set):,}')

df_fraud = df[df.label == 1].copy()
df_fraud['ym'] = df_fraud['date'].dt.to_period('M')
monthly_fraud = df_fraud.groupby('ym').size()
top_10_months = monthly_fraud.nlargest(10)
print('\nTop 10 Burst Months (48k):')
for ym, cnt in top_10_months.items():
    print(f'  {ym}: {cnt}')

WINDOW_DAYS = 7
burst_records = []
for uid, grp in df_fuser.groupby('user_id'):
    dates = grp.sort_values('date')['date'].values
    if len(dates) < 2: continue
    max_in_window = 1
    for i in range(len(dates)):
        window_end = dates[i] + np.timedelta64(WINDOW_DAYS, 'D')
        cnt = int(np.sum(dates[i:] <= window_end))
        if cnt > max_in_window: max_in_window = cnt
    burst_records.append({'user_id': uid, 'total': len(grp), 'max7': max_in_window})

burst_df = pd.DataFrame(burst_records)
heavy_burst = burst_df[(burst_df['max7'] >= 3) & (burst_df['max7']/burst_df['total'] >= 0.5)]
print(f'\nHeavy Burst Users (48k): {len(heavy_burst)}')

burst_feat = burst_df.set_index('user_id')[['max7']]
df_with_burst = df.merge(burst_feat, left_on='user_id', right_index=True, how='left').fillna(0)
print(f'\nFeature Means (48k):')
print(f"  Fraud user_burst_max7d mean: {df_with_burst.loc[df_with_burst.label==1, 'max7'].mean():.3f}")
print(f"  Normal user_burst_max7d mean: {df_with_burst.loc[df_with_burst.label==0, 'max7'].mean():.3f}")
