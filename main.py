import os
import sys

import dataloader.update_tif_name as update_tif_name
from FRET_segmentation.segmentation import SegmentationModel
from analysis.PCAModel import PCAModel


# A549细胞
A199_A549_2 = "D:\\data\\20240716\\A199-A549-2"
A_A549_2 = "D:\\data\\20240716\\A-A549-2"
C_A549_2 = "D:\\data\\20240716\\C-A549-2"

A199_A549_4 = "D:\\data\\20240716\\A199-A549-4"
A_A549_4 = "D:\\data\\20240716\\A-A549-4"
C_A549_4 = "D:\\data\\20240716\\C-A549-4"

A199_A549_6 = "D:\\data\\20240716\\A199-A549-6"
A_A549_6 = "D:\\data\\20240716\\A-A549-6"
C_A549_6 = "D:\\data\\20240716\\C-A549-6"

A199_A549_14 = "D:\\data\\20240716\\A199-A549-14"
A_A549_14 = "D:\\data\\20240716\\A-A549-14"
C_A549_14 = "D:\\data\\20240716\\C-A549-14"


# huh7细胞
A199_Huh7_4 = "D:\\data\\20240716\\A199-Huh7-3.5"
A_Huh7_4 = "D:\\data\\20240716\\A-Huh7-3.5"
C_Huh7_4 = "D:\\data\\20240716\\C-Huh7-3.5"

image_path = A199_A549_4

# 修改明场图像名字
update_tif_name.chang_BF_name(image_path)

# 进行图像细胞分割
segmentationModel = SegmentationModel(image_path)
segmentationModel.start()

# 进行Huh7图像分析
project_dir = os.path.dirname(sys.argv[0])
# need_ED_features = ['Intensity_MeanIntensity_ED']
FRET_hypothesis_feature = {
    'A': 1,
    'C': 0,
    'A199': 1
}
#
# pcaModel = PCAModel("20240716_Huh7_4h",
#                     root=project_dir,
#                     reduce_model='PCA',
#                     standard='MinMaxScaler',
#                     ED_features=None,
#                     FRET_hypothesis_feature=FRET_hypothesis_feature)
# pcaModel.start('data/csv/20240716_Huh7_BF_4h.csv', 'data/csv/20240716_Huh7_FI_4h.csv')

# 进行A549图像分析
# pcaModel = PCAModel("20240716_A549_4h",
#                     root=project_dir,
#                     reduce_model='PCA',
#                     standard='MinMaxScaler',
#                     ED_features=None,
#                     FRET_hypothesis_feature=FRET_hypothesis_feature)
# pcaModel.start('data/csv/20240716_A549_BF_4h.csv', 'data/csv/20240716_A549_FI_4h.csv')