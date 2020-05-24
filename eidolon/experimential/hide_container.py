import tensorflow as tf

from eidolon import train, loader
from eidolon.model import pixel
from eidolon import loss_util, train_tool
from eidolon.experimential.wm_container import ExtractInvisible
import numpy as np


import os
import sys
import importlib

def create_fuck_message(batch):

    temp=np.zeros([batch,128,128,3])

    for i in range(batch):
        rand = np.random.randint(0,2,(64))
        rand=rand*2-1
        rand=np.repeat(rand,256)
        rand=np.reshape(rand,[128,128])

        temp[i,:,:,0]=rand
        temp[i,:,:,1]=rand
        temp[i,:,:,2]=rand

    temp=tf.cast(temp, tf.float32)

    return temp


class HideContainer(train.Container):

    def on_prepare(self):

        # 载入数据集
        self.prepare_image_dataset()

        # 创建模型
        self.model =  pixel.UNet(input_shape=[128,128,6],
                              high_performance_enable=self.config_loader.high_performance)
        self.hide_model=ExtractInvisible([128,128,6])
        self.hide_model.summary()

        self.reveal_model=pixel.UNet(input_shape=[128,128,6],
                              high_performance_enable=self.config_loader.high_performance)
        
        # 设置优化器
        optimizer = tf.keras.optimizers.Adam(1e-4)

        # 注册模型与优化器
        self.register_model_and_optimizer(
            optimizer, {"model": self.model, "hide_model":self.hide_model, "reveal_model":self.reveal_model}, "optimizer")
        # self.register_model_and_optimizer(
        #     optimizer, {"model": self.model}, "optimizer")
        print("Initial model and optimizer....")

        #风格
        style_image=train_tool.read_image("./style.jpg",128,128, change_scale=False)
        style_image=tf.cast(style_image, tf.float32)
        style_image=style_image/127.5-1
        style_batch=np.zeros([self.config_loader.batch_size,128,128,3])


        for i in range(self.config_loader.batch_size):
            style_batch[i,:,:,:]=style_image[0]

        self.style_batch=style_batch
       
        # 注册需要记录的损失名称
        self.register_display_loss(["loss","s","h1","h2","r"])

        # 调用父类
        super(HideContainer, self).on_prepare()

    def before_train_batch(self):

        #消息
        return create_fuck_message(self.config_loader.batch_size)




    def compute_loss_function(self, each_batch, extra_batch_data):
        """
        损失函数
        Returns:
            [type] -- [description]
        """

        # 分离输入与标签
        inputs, labels = each_batch

        message=extra_batch_data

        #常规
        outputs=self.model(tf.concat([inputs, self.style_batch], axis=-1), training=True)

        #隐藏
        hidden_message=self.hide_model(tf.concat([message, self.style_batch], axis=-1))
        stego=self.model(tf.concat([inputs, hidden_message], axis=-1),training=True)

        #提取
        message_output=self.reveal_model(tf.concat([stego,tf.zeros_like(stego)],axis=-1),training=True)
        

        # 任务损失
        style_loss=loss_util.pixel_loss(outputs, labels)

        #隐藏损失
        hide1_loss=loss_util.pixel_loss(hidden_message, self.style_batch)
        hide2_loss=loss_util.pixel_loss(stego, labels)

        #提取
        reveal_loss=loss_util.pixel_loss(message_output, message)

        #总
        loss=style_loss+hide1_loss+hide2_loss+reveal_loss
        # loss=style_loss

        # 返回结果集
        return {"optimizer": loss}, {"loss": loss,"s":style_loss, "h1":hide1_loss,"h2":hide2_loss,"r":reveal_loss}

    def visual_function(self, each_batch, extra_batch_data):
        """
        视觉测试，在测试集上选择一个结果输出可视图像
        """
        # 生成测试结果
        # 分离输入与标签
        inputs, labels = each_batch

        message=extra_batch_data

        #常规
        outputs=self.model(tf.concat([inputs, self.style_batch], axis=-1), training=True)

        #隐藏
        hidden_message=self.hide_model(tf.concat([message, self.style_batch], axis=-1),training=True)
        stego=self.model(tf.concat([inputs, hidden_message], axis=-1),training=True)

        #提取
        message_output=self.reveal_model(tf.concat([stego,tf.zeros_like(stego)], axis=-1),training=True)
        
        # 结果图像
        image_list = [inputs, self.style_batch, outputs, message, hidden_message, stego, message_output,labels]
        # 标题
        title_list = ["IN", "STY", "OUT", "MSG", "HM", "STEGO", "MO","GT"]

        return {
            "image": {
                "image_list": image_list,
                "title_list": title_list,
            }
        }

    
