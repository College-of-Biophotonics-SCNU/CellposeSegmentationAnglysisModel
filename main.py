import os
import sys

import dataloader.update_tif_name as update_tif_name
from FRET_segmentation.segmentation import SegmentationModel
from analysis.PCAModel import PCAModel

root_1 = "D:\\data\\20240716\\A199-A549-4"
root_2 = "D:\\data\\20240716\\A-A549-4"
root_3 = "D:\\data\\20240716\\C-A549-4"

A199_Huh7_4 = "D:\\data\\20240716\\A199-Huh7-3.5"
A_Huh7_4 = "D:\\data\\20240716\\A-Huh7-3.5"
C_Huh7_4 = "D:\\data\\20240716\\C-Huh7-3.5"


# 修改明场图像名字
# update_tif_name.chang_BF_name(C_Huh7_4)

# 进行图像细胞分割
# segmentationModel = SegmentationModel(C_Huh7_4)
# segmentationModel.start()

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
pcaModel = PCAModel("20240716_A549_4h",
                    root=project_dir,
                    reduce_model='PCA',
                    standard='MinMaxScaler',
                    ED_features=None,
                    FRET_hypothesis_feature=FRET_hypothesis_feature)
pcaModel.start('data/csv/20240716_A549_BF_4h.csv', 'data/csv/20240716_A549_FI_4h.csv')