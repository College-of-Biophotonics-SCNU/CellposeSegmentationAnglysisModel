import matplotlib.pyplot as plt
import pandas as pd


def pie():
    # 外层饼图数据
    outer_labels = ['ED features', 'FRET FI features', 'BF features']
    outer_sizes = [27, 241, 211]

    # 内层饼图数据
    inner_labels = ['Intensity', 'Intensity distribution', 'Area Shape',
                    'Texture', 'Intensity Correlation', 'Area Shape', 'Texture']
    inner_sizes = [15, 12, 55, 156, 30, 55, 156]

    # 颜色设置
    outer_colors = ['#ff9999', '#66b3ff', '#99ff99']
    inner_colors = ['#ffcc99', '#ffff99', '#ccff99', '#99ccff',
                    '#ff99ff', '#c299ff', '#ff6699', '#99ff66', '#6699ff', '#ff9966']

    # 绘制外层饼图
    fig, ax = plt.subplots()
    wedges_outer, texts_outer, autotexts_outer = ax.pie(outer_sizes, labels=outer_labels, colors=outer_colors, radius=1,
           wedgeprops=dict(width=0.3, edgecolor='w'),
           autopct='',
           textprops={'fontsize': 12},
           startangle=90)

    # 绘制内层饼图
    ax2 = plt.subplot(111, aspect='equal')
    wedges_inner, texts_inner, autotexts_inner = ax2.pie(inner_sizes, colors=inner_colors,
                                                         startangle=90,
                                                         autopct='%1.1f%%',
                                                         radius=0.7,
                                                         pctdistance=0.8,
                                                         wedgeprops=dict(width=0.3, edgecolor='w'),
                                                         labeldistance=1.0)
    plt.legend(wedges_inner, inner_labels, loc='lower left', bbox_to_anchor=(1, 0))

    # 添加标题
    plt.title('Features Distribution')

    # 显示图形
    plt.show()


def scatter(xlsx_path):
    # 读取 Excel 表格
    df = pd.read_excel(xlsx_path)

    # 定义不同标签对应的颜色
    colors = {'FRET-BF': 'red', 'FRET': 'blue', 'BF': 'green', 'control': 'purple'}

    # 绘制散点图
    for label, group in df.groupby('label'):
        plt.scatter(group['x'], group['y'], c=colors[label], label=label)

    plt.xlabel('BF features reduce')
    plt.ylabel('FRET hypothesis features reduce')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # pie()
    scatter('../data/xlsx/data.xlsx')