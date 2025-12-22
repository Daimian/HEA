import pandas as pd
import matplotlib.pyplot as plt
import os

root = os.getcwd()
# file_path = f'{root}/performance_record.txt'
# data = pd.read_csv(file_path, sep=r'\s+', engine='python')
B0 = pd.read_csv('Quat_30FI_06A_B0_01.csv')
Mm = pd.read_csv('Quat_30FI_06A_Mm_00.csv')
Tc = pd.read_csv('Quat_30FI_06A_Tc_00.csv')
# 绘制图形
fig, ax = plt.subplots(figsize=(10, 6))

# # 绘制 current_max_value 折线
# ax.plot(data['number_of_samples'], data['current_best_value'], label='Max Value in train set', marker='o', lw=3, markersize=10)

# ax.plot(data['number_of_samples'], data['best_value_of_this_iteration'], label='Max value found in each iteration', marker='o', lw=3, markersize=10)
ax.plot(B0['number_of_samples'], B0['best_value_of_this_iteration']*100, label='Bulk modulus (best)', marker='o', lw=1.5, markersize=5, c='b', linestyle='dashed')
ax.plot(Mm['number_of_samples'], Mm['best_value_of_this_iteration']*100, label='Magnetic moment (best)', marker='o', lw=1.5, markersize=5, c='r', linestyle='dashed')
ax.plot(Tc['number_of_samples'], Tc['best_value_of_this_iteration']*100, label='Curie temperature (best)', marker='o', lw=1.5, markersize=5, c='g', linestyle='dashed')

# 绘制 mean_value_of_this_iteration 折线及其置信区间
# ax.errorbar(data['number_of_samples'], data['mean_value_of_this_iteration'],
#                     yerr=data['std_of_this_iteration'], label='Mean Value of each iteration with Confidence Interval', marker='o', lw=3, capsize=5, markersize=10)
ax.errorbar(B0['number_of_samples'], B0['mean_value_of_this_iteration']*100,
                    yerr=B0['std_of_this_iteration']*100, label='Bulk modulus (CI)', marker='o', lw=3, capsize=5, markersize=10, c='b')
ax.errorbar(Mm['number_of_samples'], Mm['mean_value_of_this_iteration']*100,
                    yerr=Mm['std_of_this_iteration']*100, label='Magnetic moment (CI)', marker='o', lw=3, capsize=5, markersize=10, c='r')
ax.errorbar(Tc['number_of_samples'], Tc['mean_value_of_this_iteration']*100,
                    yerr=Tc['std_of_this_iteration']*100, label='Curie temperature (CI)', marker='o', lw=3, capsize=5, markersize=10, c='g')

# 添加标签和标题
ax.set_xlabel('Number of Samples', size=20)
ax.set_ylabel('Relative Quantile (%)', size=20)
ax.set_title('CP-SO-BO-HEA42MagQuat', size=20)
ax.legend(prop=dict(size=14))
ax.tick_params(axis='both', which='major', labelsize=16)
#ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%d'))
#ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

# 显示图形
plt.grid(True)
plt.savefig(f'{root}/CP-SO-BO-HEA42MagQuat.png', bbox_inches='tight', dpi=150)
plt.show()
