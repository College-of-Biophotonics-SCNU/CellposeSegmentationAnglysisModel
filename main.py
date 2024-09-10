import dataloader.update_tif_name as update_tif_name
from FRET_segmentation.segmentation import SegmentationModel

root_1 = "D:\\data\\20240716\\A199-A549-4"
root_2 = "D:\\data\\20240716\\A-A549-4"
root_3 = "D:\\data\\20240716\\C-A549-4"

A199_Huh7_4 = "D:\\data\\20240716\\A199-Huh7-3.5"
A_Huh7_4 = "D:\\data\\20240716\\A-Huh7-3.5"
C_Huh7_4 = "D:\\data\\20240716\\C-Huh7-3.5"
# 修改明场图像名字
# update_tif_name.chang_BF_name(C_Huh7_4)
# 进行图像细胞分割
segmentationModel = SegmentationModel(C_Huh7_4)
segmentationModel.start()
