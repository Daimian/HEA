import pandas as pd

df = pd.read_csv("performance_record.txt", sep="\t")
print(df)

#df.to_csv('CP_SO_BO_prob_00.csv', index=False)
#df.to_csv('Quat_20FI_06A_B0_01S.csv', index=False)
#df.to_csv('Quat_20FI_13A_B0_00.csv', index=False)
#df.to_csv('Quat_30FI_06A_B0_01.csv', index=False)
df.to_csv('Quat_30FI_06A_Tc_Mm_00.csv', index=False)

