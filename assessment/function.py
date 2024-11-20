"""
药效评估函数对于 xlsx 文件进行处理计算操作
高师兄定义的药物评估函数
"""
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


class EfficacyEvaluationModel:

    def __init__(self, root, xlsx_name):
        self.hours = None
        self.treatments = None
        self.Texture_Entropy_Mean_average = {}
        self.AreaShape_FormFactor_average = {}
        self.P = {}
        self.root = root
        self.xlsx_name = xlsx_name
        self.bf_df = pd.read_excel(root + '/' + xlsx_name)

    def start(self):
        self.calculate_mean_of_texture_entropy()
        self.calculate_feature_definition_formula()

    def calculate_mean_of_texture_entropy(self):
        """
        计算纹理熵均值
        :return:
        """
        # 筛选出以 Texture_Entropy_ 开头的列名
        entropy_columns = [col for col in self.bf_df.columns if col.startswith('Texture_Entropy_')]
        # 计算每一行这些列的均值，并添加新的列保存结果
        self.bf_df['Texture_Entropy_Mean'] = self.bf_df[entropy_columns].mean(axis=1)

    def calculate_feature_definition_formula(self, time_division=True):
        """
        计算特征
        :return:
        """
        # 统计所有不同时间点
        self.hours = self.bf_df['Metadata_hour'].unique()
        self.treatments = self.bf_df['Metadata_treatment'].unique()
        # 筛选出 Metadata_treatment 为 control 的行


        for treatment in self.treatments:
            # 计算每个 control 不同时间的特征均值
            self.AreaShape_FormFactor_average[treatment] = {}
            self.Texture_Entropy_Mean_average[treatment] = {}
            treatment_df = self.bf_df[self.bf_df['Metadata_treatment'] == treatment]
            for hour in self.hours:
                self.AreaShape_FormFactor_average[treatment][hour] = treatment_df[treatment_df['Metadata_hour'] == hour]['AreaShape_FormFactor'].mean()
                self.Texture_Entropy_Mean_average[treatment][hour] = treatment_df[treatment_df['Metadata_hour'] == hour]['Texture_Entropy_Mean'].mean()
            # 计算 AreaShape_FormFactor 和 Texture_Entropy_Mean 的整体均值
            self.AreaShape_FormFactor_average[treatment]['all'] = treatment_df['AreaShape_FormFactor'].mean()
            self.Texture_Entropy_Mean_average[treatment]['all'] = treatment_df['Texture_Entropy_Mean'].mean()
        print('control 组中 AreaShape_FormFactor', self.AreaShape_FormFactor_average)
        print('control 组中 Texture_Entropy_Mean',self.Texture_Entropy_Mean_average)
        # 计算表型特征指标计算公式
        for hour in self.hours:
            self.bf_df.loc[self.bf_df['Metadata_hour'] == hour, 'P'] = self.definition_formula(hour)
        self.bf_df['P_not_divide_control'] = self.definition_formula('all')
        # 计算P在不同时间点的平均值
        self.calculate_hour_treatment()
        # 绘制对应的图像
        self.draw_box_plot('P_not_divide_control')
        self.draw_box_plot('P')
        # 保存对应的结果
        self.save_result_xlsx()
        self.save_P_value_each_xlsx()


    def definition_formula(self, hour):
        """
        特征定义公式
        :return:
        """
        hour_df = self.bf_df
        if hour != 'all':
            hour_df = self.bf_df[self.bf_df['Metadata_hour'] == hour]
        # 特征定义函数，现在使用简单的线性叠加操作
        return (hour_df['AreaShape_FormFactor']/self.AreaShape_FormFactor_average['control'][hour] - 1
                + abs(hour_df['Texture_Entropy_Mean']/self.Texture_Entropy_Mean_average['control'][hour] - 1))

    def calculate_hour_treatment(self):
        """
        计算不同时间内药物作用的表型特征均值
        """
        # 计算加药组不同时间点的均值
        for treatment in self.treatments:
            self.P[treatment] = {}
            for hour in self.hours:
                self.P[treatment][hour] = {}
                self.P[treatment][hour]['P'] = self.bf_df.loc[
                    (self.bf_df['Metadata_treatment'] == treatment) & (self.bf_df['Metadata_hour'] == hour), 'P'].mean()
                self.P[treatment][hour]['P_not_divide_control'] = self.bf_df.loc[
                    (self.bf_df['Metadata_treatment'] == treatment) & (self.bf_df['Metadata_hour'] == hour), 'P_not_divide_control'].mean()
            # 计算总体平均值
            self.P[treatment]['all'] = {}
            self.P[treatment]['all']['P'] = self.bf_df.loc[self.bf_df['Metadata_treatment'] == treatment, 'P'].mean()
            self.P[treatment]['all']['P_not_divide_control'] = self.bf_df.loc[self.bf_df['Metadata_treatment'] == treatment, 'P_not_divide_control'].mean()
        print('不同时间点不同实验组的P均值 ', self.P)


    def draw_box_plot(self, p_name):
        """
        绘制箱型图进行计算操作
        :return:
        """
        # 使用Seaborn绘制箱型图
        plt.figure(figsize=(10, 6))  # 设置图形大小
        sns.boxplot(x='Metadata_hour', y=p_name, hue='Metadata_treatment', data=self.bf_df)
        plt.title('Box Plot of P by Metadata_treatment and Metadata_hour')
        plt.xlabel('Time')
        plt.ylabel('P Values')
        plt.legend(title='treatment', loc='upper right')
        plt.savefig(self.root + '/' + p_name + '_box.jpg', dpi=300)

    def save_P_value_each_xlsx(self):
        """
        保存单细胞的 P value 计算结果
        :return:
        """
        self.bf_df.to_excel(self.root + "/" + self.xlsx_name, index=False)

    def save_result_xlsx(self):
        """
        保存不同时间不同实验组均值计算结果
        :return:
        """
        rows = []
        for treatment, hour_data in self.P.items():
            for hour, value in hour_data.items():
                rows.append({'Metadata_treatment': treatment,
                             'Metadata_hour': hour,
                             'AreaShape_FormFactor': self.AreaShape_FormFactor_average[treatment][hour],
                             'Texture_Entropy_Mean': self.Texture_Entropy_Mean_average[treatment][hour],
                             'P': value['P'],
                             'P_not_divide_control': value['P_not_divide_control']})

        df = pd.DataFrame(rows)

        df.to_excel(self.root + '/result.xlsx', index=False)


if __name__ == "__main__":
    xlsx_path = r'20240716_BF.xlsx'
    treatment_root = r'D:\data\20240716'
    model = EfficacyEvaluationModel(treatment_root, xlsx_path)
    model.start()