import numpy as np
import matplotlib.pyplot as plt
from skimage import io, exposure, filters, morphology
from skimage.util import img_as_float


class PretreatmentModel:
    def __init__(self, img=None):
        self.current_img = img
        self.current_result = None
        self.img_enhanced = None

    def enhance(self):
        # 图像增强 - 对比度增强
        self.img_enhanced = exposure.equalize_adapthist(self.current_img)
        return self.img_enhanced

    def gaussian(self):
        self.current_result = filters.gaussian(self.current_img, sigma=1.5)
        return self.current_result

    def correction(self):
        img_enhanced = self.enhance()
        # 背景校正 - 减去背景均值
        background = filters.gaussian(self.current_img, sigma=10)
        img_corrected = img_enhanced - background
        return img_corrected

    def normalize(self):
        img_corrected = self.correction()
        # 图像归一化
        img_normalized = img_as_float(img_corrected)
        img_normalized = exposure.rescale_intensity(img_normalized,
                                                    in_range=(np.percentile(img_normalized, 2),
                                                              np.percentile(img_normalized, 98)))
        return img_normalized

    def threshold(self):
        img_normalized = self.normalize()
        # 去除异常值 - 阈值处理
        threshold = np.mean(img_normalized) + 3 * np.std(img_normalized)
        img_thresholded = np.where(img_normalized > threshold, threshold, img_normalized)
        self.current_result = img_thresholded

    def show(self):
        # 显示预处理前后的图像
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(self.current_img)
        axes[0].set_title('Original Image')
        axes[1].imshow(self.current_result)
        axes[1].set_title('Preprocessed Image')
        plt.show()

    def save_img(self):
        io.imsave('../example_data/pretreatment_AA.tif', self.current_result)


if __name__ == "__main__":
    # 加载荧光图像
    img = io.imread('../example_data/AA.tif')
    pretreatmentModel = PretreatmentModel(img)
    pretreatmentModel.gaussian()
    pretreatmentModel.save_img()
