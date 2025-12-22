import pandas as pd
import matplotlib.pyplot as plt
import os
#import pickle

# fn =
# with open('Quat_30FI_06A_Tc_Mm_00.pickle', 'rb') as f:
#     data = pick
data = pd.read_csv('Quat_30FI_06A_Tc_Mm_00.csv')

# 绘制图形
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制 current_max_value 折线
ax.plot(data['number_of_samples'], data['current_best_value'], label='Max Value in train set', marker='o', lw=3, markersize=10)

ax.plot(data['number_of_samples'], data['best_value_of_this_iteration'], label='Max value found in each iteration', marker='o', lw=3, markersize=10)

# 绘制 mean_value_of_this_iteration 折线及其置信区间
ax.errorbar(data['number_of_samples'], data['mean_value_of_this_iteration'],
                    yerr=data['std_of_this_iteration'], label='Mean Value of each iteration with Confidence Interval', marker='o', lw=3, capsize=5, markersize=10)

# 添加标签和标题
ax.set_xlabel('Number of Samples', size=20)
ax.set_ylabel('Hypervolume', size=20)
ax.set_title('Algorithm Performance Over Iterations', size=20)
ax.legend(prop=dict(size=14))
ax.tick_params(axis='both', which='major', labelsize=16)
#ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%d'))
#ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

# 显示图形
plt.grid(True)
plt.savefig('performance_record.png', bbox_inches='tight', dpi=150)
plt.show()
