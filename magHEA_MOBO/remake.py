
import pandas as pd 
df = pd.read_csv('../MO_BO_3dmetal_15/3d_metal_data.csv')
df = df.head(350)
print(df.describe())

# df = df[(df['T_C'] < 600)]
df = df.sample(n=50, random_state=42)

print(df.describe())
df.to_csv('3d_metal_data.csv', index=False)