import tensorflow as tf

from eidolon import train, loader
from eidolon.model import pixel
from eidolon import loss_util, train_tool
import numpy as np

import os


class StegoContainer(train.Container):

    def on_prepare(self):

         # 载入数据
        self.prepare_image_dataset(load_function=loader.load_single_crop_no_rescale)

        #获取图像尺寸
        image_size=self.config_loader.config["dataset"]["image_size"]["value"]

        # 创建模型
        # self.model = basic.make_Conv_models(self.config_loader.config["dataset"]["image_size"]["value"])
        self.encoder = pixel.UNet([image_size[0],image_size[1],image_size[2]*2], high_performance_enable=self.config_loader.high_performance)
        self.decoder = pixel.UNet(image_size, high_performance_enable=self.config_loader.high_performance)
        # 设置优化器
        optimizer = tf.keras.optimizers.Adam(1e-4)
        # 注册模型与优化器
        self.register_model_and_optimizer(
            optimizer, {"encoder": self.encoder,"decoder":self.decoder}, "optimizer")
        print("Initial model and optimizer....")

        # 注册需要记录的损失名称
        self.register_display_loss(["encoder loss","decoder loss"])

        # 调用父类
        super(StegoContainer, self).on_prepare()

    def before_train_batch(self):

        for extra_batch in self.train_dataset.take(1):
            pass

        return extra_batch


    def compute_loss_function(self, cover_batch, secret_batch):

        inputs=tf.concat([cover_batch, secret_batch], axis=-1)
        stego = self.encoder(inputs, training=True)

        outputs=self.decoder(stego, training=True)

        encoder_loss=loss_util.pixel_loss(stego, cover_batch)

        decoder_loss=loss_util.pixel_loss(outputs, secret_batch)

        loss = encoder_loss+decoder_loss

        # 返回结果集
        return {"optimizer": loss}, {"encoder loss": encoder_loss, "decoder loss":decoder_loss}

    def visual_function(self, cover_batch, secret_batch):
        """
        视觉测试，在测试集上选择一个结果输出可视图像
        """
        inputs=tf.concat([cover_batch, secret_batch], axis=-1)

        stego = self.encoder(inputs, training=True)
        outputs=self.decoder(stego, training=True)

        image_list=[cover_batch, secret_batch, stego, outputs]
        title_list=["cover","secret","stego","output"]

        return image_list, title_list

