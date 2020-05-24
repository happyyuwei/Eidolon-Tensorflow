import tensorflow as tf

from eidolon import train, loader
from eidolon.model import pixel
from eidolon import loss_util, train_tool
import numpy as np

import os


class HideExtractorContainer(train.Container):
    """
    将水印提取网络隐藏进目标网络
    """


    def on_prepare(self):

         # 载入数据
        self.prepare_image_dataset()

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
        self.register_display_loss(["encoder loss","decoder loss","task loss"])

        # 调用父类
        super(HideExtractorContainer, self).on_prepare()

    def before_train_batch(self):

        for extra_batch in self.train_dataset.take(1):
            pass

        inputs,style=extra_batch

        return inputs


    def compute_loss_function(self, task_batch, secret_batch):

        origin,target=task_batch

        # stego网络
        inputs=tf.concat([origin, secret_batch], axis=-1)
        stego = self.encoder(inputs, training=True)

        secret_outputs=self.decoder(stego, training=True)

        encoder_loss=loss_util.pixel_loss(stego, origin)

        decoder_loss=loss_util.pixel_loss(secret_outputs, secret_batch)

        #目标网络
        task_outputs = self.decoder(origin, training=True)
        task_loss=loss_util.pixel_loss(target, task_outputs)

        loss = encoder_loss+decoder_loss+task_loss

        # 返回结果集
        return {"optimizer": loss}, {"encoder loss": encoder_loss, "decoder loss":decoder_loss,"task loss":task_loss}

    def visual_function(self, task_batch, secret_batch):
        """
        视觉测试，在测试集上选择一个结果输出可视图像
        """
        origin,target=task_batch
        
        inputs=tf.concat([origin, secret_batch], axis=-1)
        stego = self.encoder(inputs, training=True)

        secret_outputs=self.decoder(stego, training=True)
        task_outputs = self.decoder(origin, training=True)

        image_list=[origin, secret_batch, stego, secret_outputs, task_outputs,target]
        title_list=["orogin","secret","stego","secret_output","task_output","target"]

        return image_list, title_list




class HideEncoderContainer(train.Container):
    """
    将水印提取网络隐藏进目标网络
    """


    def on_prepare(self):

         # 载入数据
        self.prepare_image_dataset()

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
        self.register_display_loss(["encoder loss","decoder loss","task loss"])

        self.mask=np.ones([1,128,128,3])
        self.mask[0:20,0:20]=0
        self.mask=tf.cast(self.mask, tf.float32)

        # 调用父类
        super(HideEncoderContainer, self).on_prepare()

    def before_train_batch(self):

        for extra_batch in self.train_dataset.take(1):
            pass

        inputs,style=extra_batch

        return inputs


    def compute_loss_function(self, task_batch, secret_batch):

        origin,target=task_batch

        
        # secret_batch=secret_batch*self.mask

        # stego网络
        inputs=tf.concat([origin, secret_batch], axis=-1)
        stego = self.encoder(inputs, training=True)


        secret_outputs=self.decoder(stego, training=True)

        encoder_loss=loss_util.pixel_loss(stego, origin)

        decoder_loss=loss_util.pixel_loss(secret_outputs, secret_batch)

        #目标网络
        task_outputs = self.encoder(tf.concat([origin, tf.zeros_like(origin)],axis=-1), training=True)
        task_loss=loss_util.pixel_loss(target, task_outputs)

        loss = encoder_loss+decoder_loss+task_loss

        # 返回结果集
        return {"optimizer": loss}, {"encoder loss": encoder_loss, "decoder loss":decoder_loss,"task loss":task_loss}

    def visual_function(self, task_batch, secret_batch):
        """
        视觉测试，在测试集上选择一个结果输出可视图像
        """
        origin,target=task_batch
        
        inputs=tf.concat([origin, secret_batch], axis=-1)
        stego = self.encoder(inputs, training=True)

        secret_outputs=self.decoder(stego, training=True)
         #目标网络
        task_outputs = self.encoder(tf.concat([origin, tf.zeros_like(origin)],axis=-1), training=True)

        image_list=[origin, secret_batch, stego, secret_outputs, task_outputs,target]
        title_list=["origin","secret","stego","secret_output","task_output","target"]

        return image_list, title_list