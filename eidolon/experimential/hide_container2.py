import tensorflow as tf

from eidolon import train, loader
from eidolon.model import pixel
from eidolon import loss_util, train_tool, eval_util
from eidolon.experimential.wm_container import ExtractInvisible
import numpy as np


import os
import sys
import importlib

def create_fuck_message(batch, message_len_sqrt, img_width):

    temp = np.zeros([batch, 128, 128, 3])
    repeat_len = img_width/message_len_sqrt

    for i in range(batch):
        rand = np.random.randint(0, 2, ([message_len_sqrt, message_len_sqrt]))
        rand = rand*2-1
        rand = np.repeat(rand, repeat_len, axis=1)
        rand = np.repeat(rand, repeat_len, axis=0)

        temp[i, :, :, 0] = rand
        temp[i, :, :, 1] = rand
        temp[i, :, :, 2] = rand

    temp = tf.cast(temp, tf.float32)

    return temp


class HideContainer(train.Container):

    def on_prepare(self):

        # 载入数据集
        self.prepare_image_dataset()

        # 创建模型
        self.model =  pixel.UNet(input_shape=[128,128,6],
                              high_performance_enable=self.config_loader.high_performance)
        # self.hide_model=ExtractInvisible([128,128,6])
        # self.hide_model.summary()

        # self.reveal_model=pixel.UNet(input_shape=[128,128,6],
        #                       high_performance_enable=self.config_loader.high_performance)
        
        # 设置优化器
        optimizer = tf.keras.optimizers.Adam(1e-4)

        # 注册模型与优化器
        # self.register_model_and_optimizer(
        #     optimizer, {"model": self.model, "hide_model":self.hide_model, "reveal_model":self.reveal_model}, "optimizer")
        self.register_model_and_optimizer(
            optimizer, {"model": self.model}, "optimizer")
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
        self.register_display_loss(["loss","hide_loss","reveal_loss","style_loss"])

        # 每行信息数量
        self.row_num=8

        # 调用父类
        super(HideContainer, self).on_prepare()


    def before_train_batch(self):
        #消息
        return create_fuck_message(self.config_loader.batch_size, self.row_num, 128)


    def compute_loss_function(self, each_batch, extra_batch_data):
        """
        损失函数
        Returns:
            [type] -- [description]
        """

        # 分离输入与标签
        cover, labels = each_batch

        message=extra_batch_data

        #常规
        outputs=self.model(tf.concat([cover, self.style_batch], axis=-1), training=True)

        #隐藏
        # hidden_message=self.hide_model(tf.concat([message, self.style_batch], axis=-1))
        stego=self.model(tf.concat([cover, message], axis=-1),training=True)

        #提取
        message_output=self.model(tf.concat([stego,tf.zeros_like(stego)],axis=-1),training=True)
        
        # 任务损失
        style_loss=loss_util.pixel_loss(outputs, labels)

        #隐藏损失
        # hide1_loss=loss_util.pixel_loss(hidden_message, self.style_batch)
        hide_loss=loss_util.pixel_loss(stego, cover)

        #提取
        reveal_loss=loss_util.pixel_loss(message_output, message)

        #总
        loss=hide_loss+reveal_loss+style_loss
        # loss=style_loss

        # 返回结果集
        return {"optimizer": loss}, {"loss": loss,"hide_loss":hide_loss,"reveal_loss":reveal_loss}

    def visual_function(self, each_batch, extra_batch_data):
        """
        视觉测试，在测试集上选择一个结果输出可视图像
        """
        # 生成测试结果
        # 分离输入与标签
        cover, labels = each_batch

        message=extra_batch_data

        #常规
        outputs=self.model(tf.concat([cover, self.style_batch], axis=-1), training=True)

        #嵌入
        stego=self.model(tf.concat([cover, message], axis=-1),training=True)

        #提取
        message_output=self.model(tf.concat([stego,tf.zeros_like(stego)],axis=-1),training=True)
        
        # 结果图像
        image_list = [cover, stego, message, message_output, outputs, labels]
        # 标题
        title_list = ["COVER", "STEGO", "MSG", "MSGOUT", "OUT", "GT"]

        return {
            "image": {
                "image_list": image_list,
                "title_list": title_list,
            }
        }

    def parse_msg(self, message_tensor):
        """四维张量

        Arguments:
            message_mat {[type]} -- [description]
        """
        temp=np.mean(message_tensor, axis=3)
        message_tensor=np.zeros([self.config_loader.batch_size, 128, 128, 1])
        message_tensor[:,:,:,0]=temp

        #过滤尺寸
        message_tensor[message_tensor<0]=-1
        message_tensor[message_tensor>0]=1

        # print(tf.shape(message_tensor))
        msg=tf.nn.avg_pool2d(message_tensor,ksize=128//self.row_num, strides=128//self.row_num, padding="VALID")
        
        return msg
        




    def compute_test_metrics_function(self, each_batch, extra_batch_data):


        cover, label = each_batch

        message=extra_batch_data

        #常规
        output=self.model(tf.concat([cover, self.style_batch], axis=-1), training=True)

        #嵌入
        stego=self.model(tf.concat([cover, message], axis=-1),training=True)

        #提取
        message_output=self.model(tf.concat([stego,tf.zeros_like(stego)],axis=-1),training=True)

        #解析信息
        message_output= self.parse_msg(message_output)

        message=self.parse_msg(message)
        # print(message_output)
        # print(message)


        #评估, stego
        stego_psnr=eval_util.evaluate(stego, cover, psnr_enable=True, ssim_enable=False, ber_enable=False)
        # 信息提取
        msg_ber=eval_util.evaluate(message, message_output, psnr_enable=False, ssim_enable=False, ber_enable=True)
        # 任务psnr
        style_result=eval_util.evaluate(label, output,psnr_enable=True, ssim_enable=True, ber_enable=False)


        return {"stego_psnr": stego_psnr["psnr"], "msg_ber":msg_ber["ber"], "style_psnr": style_result["psnr"], "style_ssim":style_result["ssim"]}
