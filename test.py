import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 定义函数进行拼接或输出 C
def create_label(row):
    if row['Metadata_treatment'] == 'C':
        return 'C'
    else:
        return f"{row['Metadata_treatment']}_{row['Metadata_hour']}"


# 读取 CSV 文件
df = pd.read_csv('analysis/new_data.csv')

# 使用 apply 函数创建新列
labels = df.apply(create_label, axis=1)

# 假设要绘制某一列名为 'column_name'，对应的标签列为 'label_column'
data = df['reduced_feature']

# 定义不同标签对应的颜色
label_colors = {}
unique_labels = np.unique(labels)
for i, label in enumerate(unique_labels):
    label_colors[label] = plt.cm.tab20(i % 20)

print(unique_labels)

# 划分区间
num_bins = 50
min_value = np.min(data)
max_value = np.max(data)
bin_width = (max_value - min_value) / num_bins
bins = [min_value + i*bin_width for i in range(num_bins + 1)]

# 绘制合并的直方图
plt.hist([data[labels == label] for label in unique_labels], bins=bins,
         label=[label for label in unique_labels],
         color=[label_colors[label] if label in label_colors else 'k' for label in unique_labels])
plt.title('Combined Frequency Histogram for Different Labels')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()