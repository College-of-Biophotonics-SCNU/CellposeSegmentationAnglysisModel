"""
使用降维算法计算加药特征与未加药的区别
1. PCA-1 维度使用明场维度的特征
2. PCA-2 维度使用荧光维度的特征
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class PCAModel:
    def __init__(self, experiment="", ED_features=[]):
        self.file_list = []
        self.BF_features = []
        self.FI_features = []
        self.BF_original_df_data = {}
        self.FI_original_df_data = {}
        self.BF_df = {}
        self.FI_df = {}
        self.BF_FI_df = {}
        self.labels = []
        self.BF_suffixes = '_from_BF'
        self.FI_suffixes = '_from_FI'
        # 获取共同的主键列名
        self.primary_keys = ['ImageNumber', 'ObjectNumber', 'label']
        # 实验批次名称
        self.experiment = experiment
        self.ED_features = ED_features

    def start(self, BF_df_data_path, FI_df_data_path):
        self.pre_data_loader(BF_df_data_path, FI_df_data_path)
        self.standardization()
        self.merge_data()
        if len(self.ED_features) == 0:
            self.reduce_analysis_FI_BF_features()
        self.reduce_analysis_ED_BF_features(self.ED_features)

    def pre_data_loader(self, BF_df_data, FI_df_data):
        """
        数据的读取
        :return:
        """
        BF_df_data = pd.read_csv(BF_df_data)
        FI_df_data = pd.read_csv(FI_df_data)
        self.labels = BF_df_data['Metadata_treatment']
        BF_df_data['label'] = BF_df_data['Metadata_treatment']
        FI_df_data['label'] = FI_df_data['Metadata_treatment']
        BF_original_df_data = BF_df_data.drop(
            columns=[col for col in BF_df_data.columns if col.startswith('Metadata_')])
        FI_original_df_data = FI_df_data.drop(
            columns=[col for col in FI_df_data.columns if col.startswith('Metadata_')])
        print("BF原始数据大小为", BF_original_df_data.shape)
        print("FI原始数据大小为", FI_original_df_data.shape)
        # 删除 null 值得行数据
        self.BF_original_df_data = BF_original_df_data.dropna()
        self.FI_original_df_data = FI_original_df_data.dropna()
        self.BF_original_df_data.reset_index(drop=True, inplace=True)
        self.FI_original_df_data.reset_index(drop=True, inplace=True)

    def pre_data_del_outlier(self):
        # 删除异常值
        pass

    def standardization(self):
        """
        数据标准化
        :return:
        """
        print("清除了null值后的特征BF特征矩阵有", self.BF_original_df_data.shape)
        print("清除了null值后的特征FI特征矩阵有", self.FI_original_df_data.shape)
        # 提取要进行归一化的列
        BF_data_to_normalize = self.BF_original_df_data.drop(columns=self.primary_keys)
        self.BF_features = BF_data_to_normalize.columns
        FI_data_to_normalize = self.FI_original_df_data.drop(columns=self.primary_keys)
        self.FI_features = FI_data_to_normalize.columns

        # 创建归一化器对象
        scaler = StandardScaler()
        # 对数据进行归一化
        BF_normalized_data = scaler.fit_transform(BF_data_to_normalize)
        FI_normalized_data = scaler.fit_transform(FI_data_to_normalize)
        # 将归一化后的数据与不需要归一化的列合并
        self.BF_df = pd.concat([self.BF_original_df_data[['ImageNumber', 'ObjectNumber', 'label']],
                                pd.DataFrame(BF_normalized_data, columns=self.BF_features)], axis=1)
        self.FI_df = pd.concat([self.FI_original_df_data[['ImageNumber', 'ObjectNumber', 'label']],
                                pd.DataFrame(FI_normalized_data, columns=self.FI_features)], axis=1)

        print("BF归一化后的特征矩阵大小为", self.BF_df.shape)
        print("FI归一化后的特征矩阵大小为", self.FI_df.shape)

    def merge_data(self):
        # 合并两个数据帧concat([self.BF_df, df4], axis=1, join='inner')
        self.BF_FI_df = pd.merge(self.BF_df, self.FI_df, on=self.primary_keys, how='inner',
                                 suffixes=(self.BF_suffixes, self.FI_suffixes))
        self.labels = self.BF_FI_df['label']
        print("合并后的数据大小为", self.BF_FI_df.shape)

    def reduce_analysis_FI_BF_features(self, is_need_reduce_separate=True):
        """
        降维分析操作
        :return:
        """
        # 分别进行降维 降维成为1维
        pca_BF = PCA(n_components=1)
        pca_FI = PCA(n_components=1)
        BF_transformed = pca_BF.fit_transform(self.BF_FI_df[[
            feature + self.BF_suffixes if feature not in self.BF_FI_df.columns else feature for feature in
            self.BF_features]])
        FI_transformed = pca_FI.fit_transform(self.BF_FI_df[[
            feature + self.FI_suffixes if feature not in self.BF_FI_df.columns else feature for feature in
            self.FI_features]])
        pca_point = np.hstack((BF_transformed.reshape(-1, 1), FI_transformed.reshape(-1, 1)))
        print(BF_transformed[0])
        print(FI_transformed[0])
        print(pca_point[0])
        print("BF-FI特征降维后的点矩阵大小", pca_point.shape)
        # 绘制图像
        # 绘制散点图
        unique_labels = np.unique(self.labels)
        colors = cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        plt.xlabel('BF features reduce')  # 设置横坐标标签
        plt.ylabel('FI features reduce')  # 设置纵坐标标签

        for i, label in enumerate(unique_labels):
            indices = np.where(self.labels == label)
            plt.scatter(pca_point[indices, 0], pca_point[indices, 1], c=[colors[i]], label=label)
        plt.legend(loc='upper right')
        plt.savefig('../data/result/jpg/' + "pca_BF_FI_" + self.experiment + ".jpg", dpi=300)
        plt.show()

    def reduce_analysis_ED_BF_features(self, ED_features):
        """
        降维分析操作
        :return:
        """
        # 分别进行降维 降维成为1维
        pca_BF = PCA(n_components=1)
        pca_FI = PCA(n_components=1)
        BF_transformed = pca_BF.fit_transform(self.BF_FI_df[[feature + self.BF_suffixes
                                                             if feature not in self.BF_FI_df.columns
                                                             else feature for feature in self.BF_features]])
        FI_transformed = pca_FI.fit_transform(self.BF_FI_df[[feature + self.FI_suffixes
                                                             if feature not in self.BF_FI_df.columns
                                                             else feature for feature in self.ED_features]])
        pca_point = np.hstack((BF_transformed.reshape(-1, 1), FI_transformed.reshape(-1, 1)))
        print(BF_transformed[0])
        print(FI_transformed[0])
        print(pca_point[0])
        print("BF-FI特征降维后的点矩阵大小", pca_point.shape)
        # 绘制图像
        # 绘制散点图
        unique_labels = np.unique(self.labels)
        colors = cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        plt.xlabel('BF features reduce')  # 设置横坐标标签
        plt.ylabel('ED features reduce')  # 设置纵坐标标签

        for i, label in enumerate(unique_labels):
            indices = np.where(self.labels == label)
            plt.scatter(pca_point[indices, 0], pca_point[indices, 1], c=[colors[i]], label=label)
        plt.legend(loc='upper right')
        plt.savefig('../data/result/jpg/' + "pca_BF_ED_mean_" + self.experiment + ".jpg", dpi=300)
        plt.show()


if __name__ == '__main__':
    # 'Intensity_MaxIntensity_ED',
    # 'Intensity_MedianIntensity_ED'
    need_ED_features = ['Intensity_MeanIntensity_ED']
    pcaModel = PCAModel("20240716_A549_4h", ED_features=need_ED_features)
    pcaModel.start('../data/csv/20240716_A549_BF_4h.csv', '../data/csv/20240716_A549_FI_4h.csv')
