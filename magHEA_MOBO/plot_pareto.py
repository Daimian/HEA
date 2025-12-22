import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('3d_metal_data.csv')
columns_to_normalize = ['M_s', 'T_C', 'Pugh', 'Cauchy']
data = df[columns_to_normalize].apply(
    lambda x: (x - x.min()) / (x.max() - x.min())
)
data['pareto'] = np.sqrt(data['M_s']**2 + data['T_C']**2 + data['Pugh']**2 + data['Cauchy']**2)

def split_dataframe_by_rows(df, chunk_size):
    return [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

# Split into sub-DataFrames with 10 rows each
chunk_size = 10
sub_dfs = split_dataframe_by_rows(data, chunk_size)

# markers = ['o', 's', '^', 'D']
# _, ax = plt.subplots()

# ax.plot(data.index, data['pareto'])
means = []
stds = []
for i, sub_df in enumerate(sub_dfs):
    means.append(sub_df['pareto'].mean())
    stds.append(sub_df['pareto'].std())
x = range(len(sub_dfs))

fig, axes = plt.subplots(nrows=2, sharex=True)
axes[0].plot(x, means)
#print(means)
axes[1].plot(x, stds)
    # ax.plot(sub_df.index, sub_df['pareto'])
    # ax.plot(i, sub_df['pareto'].mean())
# for i, col in enumerate(data.columns):
#     ax.plot(data.index, data[col], label=col, 
#             marker=markers[i], markersize=7)

# Add labels and legends
# ax.set_xlabel('Iterations')
# ax.set_ylabel('Pareto')
# ax.legend()
plt.savefig('MO_BO_3d_metal_Pareto.png', bbox_inches='tight', dpi=800)