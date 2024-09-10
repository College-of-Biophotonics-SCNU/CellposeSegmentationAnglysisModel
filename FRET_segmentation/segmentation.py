import matplotlib.pyplot as plt
from cellpose import models
from skimage import io, filters
import tifffile
import numpy as np
import os

target_files = ['AA.tif', 'DD.tif', 'DA.tif']


class SegmentationModel:
    def __init__(self, root=None, img=None, diameter=200):
        self.matching_sub_folder_paths = []
        self.model = models.Cellpose(gpu=True)
        self.root = root
        self.current_mask = None
        self.current_img = img
        self.diameter = diameter

    def start(self):
        self.dataloader()
        for sub_folder_path in self.matching_sub_folder_paths:
            print("处理 ===> ", sub_folder_path)
            image1 = tifffile.imread(os.path.join(sub_folder_path, "AA.tif"))
            image2 = tifffile.imread(os.path.join(sub_folder_path, "DA.tif"))
            image3 = tifffile.imread(os.path.join(sub_folder_path, "DD.tif"))
            combined_image = np.stack((image1, image2, image3), axis=-1)
            self.mask_image(combined_image)
            # 保存图像
            io.imsave(os.path.join(sub_folder_path, 'mask_img.tif'), self.current_mask)

    def dataloader(self):
        """
        遍历文件下的每个路径，主要是获取 AA、DA、DD 通道数据
        :return:
        """
        for root, dirs, files in os.walk(self.root):
            for sub_folder in dirs:
                sub_folder_path = os.path.join(root, sub_folder)
                for target_file in target_files:
                    if os.path.exists(os.path.join(sub_folder_path, target_file)):
                        self.matching_sub_folder_paths.append(sub_folder_path)
                        break

    def gaussian(self):
        """
        预处理操作，进行高斯模糊处理
        :return:
        """
        return filters.gaussian(self.current_img, sigma=1.5)

    def mask_image(self, img):
        self.current_img = img
        img = self.gaussian()
        self.current_mask, flows, styles, diams = self.model.eval(img,
                                                                  diameter=self.diameter,
                                                                  channels=[2, 1, 0],
                                                                  resample=True)
        return self.current_mask, flows, styles, diams

    def show_mask_image(self):
        fig = plt.figure(figsize=(12, 5))
        ax1 = fig.add_subplot(121)
        ax1.imshow(self.current_img)
        ax1.set_title('Original Image')
        ax2 = fig.add_subplot(122)
        ax2.imshow(self.current_mask)
        ax2.set_title('Segmented Masks')
        plt.show()


if __name__ == "__main__":
    # image1 = tifffile.imread("../example_data/AA.tif")
    # image2 = tifffile.imread("../example_data/DA.tif")
    # image3 = tifffile.imread("../example_data/DD.tif")
    # combined_image = np.stack((image1, image2, image3), axis=-1)
    segmentationModel = SegmentationModel(root='D:\\data\\20240716\\A199-A549-4')
    segmentationModel.start()
    # segmentationModel.show_mask_image()
