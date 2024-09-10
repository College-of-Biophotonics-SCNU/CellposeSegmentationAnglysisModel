import dataloader.update_tif_name as update_tif_name
from FRET_segmentation.segmentation import SegmentationModel

root_1 = "D:\\data\\20240716\\A199-A549-4"
root_2 = "D:\\data\\20240716\\A-A549-4"
root_3 = "D:\\data\\20240716\\C-A549-4"
# 修改明场图像名字
# update_tif_name.chang_BF_name(root_3)
# 进行图像细胞分割
segmentationModel = SegmentationModel(root_3)
segmentationModel.start()
