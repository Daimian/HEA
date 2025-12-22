import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('3d_metal_data.csv')
columns_to_normalize = ['M_s', 'T_C', 'Pugh', 'Cauchy']
data = df[columns_to_normalize].apply(
    lambda x: (x - x.min()) / (x.max() - x.min())
)
markers = ['o', 's', '^', 'D']
_, ax = plt.subplots()

ax.plot()

for i, col in enumerate(data.columns):
    ax.plot(data.index, data[col], label=col, 
            marker=markers[i], markersize=7)

# Add labels and legends
ax.set_xlabel('Samples')
ax.set_ylabel('Normalized Objectives')
ax.legend()
plt.savefig('MO_BO_3d_metal_4Obj.png', bbox_inches='tight', dpi=800)