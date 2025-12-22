import pandas as pd

df = pd.read_csv('Quat_30FI_Tc_Mm.csv')

condition = (df['mag']>=1.12602697) & (df['TC']>=752.42168)
print(df[condition])
