import pandas as pd
df = pd.read_parquet('dataset/features/features_daily.parquet')
print(f"START: {df.index.min()}")
print(f"END: {df.index.max()}")
