from eidolon.pixel_container import PixelContainer
from CriminalArt import load_celebA

import numpy as np
import tensorflow as tf


class FaceContainer(PixelContainer):

    def __init__(self, config_loader):
        # 调用父类
        super(FaceContainer, self).__init__(config_loader)

    def on_prepare_dataset(self):
        # 载入数据集
        train_dataset = load_celebA.load_dataset(
            self.config_loader, is_training=True)
        test_dataset = load_celebA.load_dataset(
            self.config_loader, is_training=False)

        # 注册数据
        self.register_dataset(train_dataset, test_dataset)

    def compute_loss(self, image, label):

        # 传入的label包括label图像和向量
        _, mask = label

        shape = tf.shape(mask)

        # 加入噪声
        mask = mask + tf.random.normal(shape)

        result_set = super(FaceContainer, self).compute_loss(mask, image)

        return result_set

    def test_metrics(self, loss_set):
        """
        覆盖原始的pnsr与ssim，无需测量
        """
        return loss_set

    def test_visual(self):
        """
        视觉测试，在测试集上选择一个结果输出可视图像
        """
        # 测试可视化结果
        for test_input, test_target in self.test_dataset.take(1):

            _, mask = test_target

            # 生成测试结果
            predicted_image = self.generator(mask, training=True)

        # 排成列表
        image_list = [mask, test_input, predicted_image]
        title_list = ["IN", "GT", "PR"]
        return image_list, title_list
