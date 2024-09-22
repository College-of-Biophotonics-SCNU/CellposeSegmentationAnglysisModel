import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler


# 定义函数进行拼接或输出 C
def create_label(row):
    if row['Metadata_treatment'] == 'C':
        return 'C'
    else:
        return f"{row['Metadata_treatment']}_{row['Metadata_hour']}"


def hist_all_data_analysis(data):
    # 计算数据的最大值和最小值
    min_value = np.min(data)
    max_value = np.max(data)

    # 划分区间
    num_bins = 50
    bin_width = (max_value - min_value) / num_bins
    bins = [min_value + i * bin_width for i in range(num_bins + 1)]

    # 绘制直方图
    plt.hist(data, bins=bins)
    plt.xlabel('Reduced dimensional features')
    plt.ylabel('Frequency')
    plt.show()


def hist_different_label_feature_analysis(df):
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
    bins = [min_value + i * bin_width for i in range(num_bins + 1)]

    # 绘制合并的直方图
    plt.hist([data[labels == label] for label in unique_labels], bins=bins,
             label=[label for label in unique_labels],
             color=[label_colors[label] if label in label_colors else 'k' for label in unique_labels])
    plt.xlabel('Reduced dimensional features')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


class ReduceOneDimensionModel:
    def __init__(self, file_path, cell_line):
        # 读取 Excel 文件
        df = pd.read_csv(file_path)
        self.cell_line = cell_line  # 细胞系名称
        self.labels = df['Metadata_treatment'].unique()  # 不同干扰标签
        self.treatment_data = {}  # 不同干扰的数据
        self.pca = PCA(n_components=1)
        for label in self.labels:
            self.treatment_data[label] = df[df['Metadata_treatment'] == label]

    def start(self, is_variance=False):
        self.process_data_with_reduce(pd.concat([self.treatment_data['C'], self.treatment_data['A']]))

    def process_data_with_reduce(self, data, columns_to_exclude=None, is_variance=False):
        # 删除确实样本的值
        if columns_to_exclude is None:
            # 指定不需要归一化的列名
            columns_to_exclude = ['Metadata_hour', 'Metadata_treatment', 'ImageNumber', 'ObjectNumber']
        data = data[~data.isin([0]).any(axis=1)]
        data = data.dropna(axis=1)
        data.reset_index(drop=True, inplace=True)

        # 归一化操作 Min-Max 归一化
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(data.drop(columns_to_exclude, axis=1))

        # 特征选择 方差筛选 设置方差阈值为 0.2
        if is_variance:
            selector = VarianceThreshold(threshold=0.2)
            filtered_data = selector.fit_transform(normalized_data)
        else:
            filtered_data = normalized_data
        # 特征降维
        data_reduced = self.pca.fit_transform(filtered_data)

        # 将降维后的数据转换为 DataFrame
        reduced_df = pd.DataFrame(data_reduced, columns=['reduced_feature'])

        # 合并保留的数据和降维后的数据
        result_df = pd.concat([data[columns_to_exclude], reduced_df], axis=1)

        # 将结果保存到新的 Excel 文件
        result_df.to_csv('new_data.csv', index=False)


if __name__ == '__main__':
    # reduce = ReduceOneDimensionModel('../data/csv/20240716_A549_ED.csv', 'A549')
    # reduce.start(is_variance=True)
    # 读取 CSV 文件
    df = pd.read_csv('reduce_A549_A_ED_features.csv')
    # 假设要绘制某一列名为 'column_name' 的频数直方图
    data = df['reduced_feature']
    # hist_all_data_analysis(data)
    hist_different_label_feature_analysis(df)
