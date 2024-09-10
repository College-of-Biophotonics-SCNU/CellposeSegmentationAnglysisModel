import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 假设外貌特征数据
appearance_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
appearance_labels = np.array(['A', 'B', 'C'])

# 假设健康情况特征数据
health_data = np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]])
health_labels = np.array(['X', 'Y', 'Z'])

# 对外貌特征进行 PCA 降维到一维
pca_appearance = PCA(n_components=1)
appearance_transformed = pca_appearance.fit_transform(appearance_data)

# 对健康情况特征进行 PCA 降维到一维
pca_health = PCA(n_components=1)
health_transformed = pca_health.fit_transform(health_data)

# 组合降维后的特征作为二维数据
combined_data = np.vstack((appearance_transformed.reshape(-1, 1), health_transformed.reshape(-1, 1))).T
combined_labels = np.hstack((appearance_labels, health_labels))

# 绘制散点图
unique_labels = np.unique(combined_labels)
colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
for i, label in enumerate(unique_labels):
    indices = np.where(combined_labels == label)
    plt.scatter(combined_data[indices, 0], combined_data[indices, 1], c=[colors[i]], label=label)

plt.xlabel('Appearance Dimension')
plt.ylabel('Health Dimension')
plt.legend()
plt.show()