"""
该脚本定义水印训练，继承至pixel-to-pixel
"""
import tensorflow as tf
import numpy as np
import random

from eidolon import pixel_container
from eidolon.config import ArgsParser
from eidolon import train_tool
from eidolon import loss_util

from WMNetv2.extractor import Extractor
from model_use import EncoderDecoder



class WMContainer(pixel_container.PixelContainer):

    def __init__(self, config_loader):
        # 调用父类
        super(WMContainer, self).__init__(config_loader)

    def on_prepare(self):
        """
        准备阶段，完成以下事宜：
        1. 创建水印提取网络
        2. 加载解码器
        3. 调用父类的方法，该方法完成以下事宜：
            1. 加载数据集
            2. 创建网络与优化器
            3. 将网络与优化器注册到父类中，以便自动保存
            4. 调用父类on_prepare
        """
        """
        This will be invokde in initial process
        """
        print("This container is for training the invisible watermark.")

        # lambdas in loss
        self.lambda_wm_positive = self.config_loader.lambda_array[0]
        self.lambda_wm_negitive = self.config_loader.lambda_array[1]
        # lambdas print
        print("lp={}, ln={}".format(
            self.lambda_wm_positive, self.lambda_wm_negitive))

        # 解析所有配置
        args_parser = ArgsParser(self.config_loader.callback_args)

        # 解码器
        self.decoder_path = args_parser.get("decoder")

        # training noise attack
        self.noise_attack = args_parser.get("noise") in ("True", "true")
        print("noise attack:{}".format(self.noise_attack))

        # 训练随机裁剪增加水印裁剪攻击能力
        self.crop_attack = args_parser.get("crop") in ("True", "true")
        print("crop attack:{}".format(self.crop_attack))

        # init extract network
        print("initial WM extract network")
        self.extractor = Extractor(
            self.config_loader.config["dataset"]["image_size"]["value"])
        # 注册
        self.model_map["extractor"] = self.extractor

        # load watermark
        self.watermark_target = train_tool.read_image(
            "./watermark.png", self.config_loader.image_width, self.config_loader.image_height, change_scale=True)
        print("load watermark successfully...")

        # create negitive. if no watermark, a 1 matrix will be out
        self.negitive_target = tf.zeros(shape=[1, 128, 128, 3])*1
        print("create negative watermark successfully...")

        # the pretrained encoder-decoder model
        # @update 2019.11.27
        # @author yuwei
        # 修复相对路径bug,否则无法载入模型
        self.decoder_model = EncoderDecoder(self.decoder_path)
        print("load decoder successfully...")

        # 调用父类
        super(WMContainer, self).on_prepare()

    def compute_loss(self, input_image, target):
        """
        计算损失函数，继承父类，在此基础上加上水印损失
        """
        result_set = super(WMContainer, self).compute_loss(input_image, target)

        # no attack
        gen_output = result_set["gen_output"]

        ext_input = gen_output

        # 攻击pipline
        # create watermark
        if self.noise_attack == True:
            # create normal noise sigma from 0-0.4
            sigma = random.random()*0.4
            # noise attack， the sigma is 0-0.4, mean is 0
            normal_noise = np.random.normal(0, scale=sigma, size=[128, 128, 3])
            # 添加噪声
            ext_input = gen_output+normal_noise

        if self.crop_attack == True:
            # 创建掩码
            crop_mask = np.ones([1, 128, 128, 3], dtype=np.float32)
            # 裁剪长度为0-50个像素宽度
            crop_width = np.random.randint(0, 50)
            crop_mask[:, :, 0:crop_width, :] = 0
            # 裁剪
            ext_input = tf.multiply(gen_output, crop_mask)+crop_mask-1

        # extract the gen output (with watermark)=>watermark
        extract_watermark = self.extractor(ext_input, training=True)

        # negitive samples
        extract_negitive = self.extractor(target, training=True)

        # watermark error, close to watermark target
        watermark_possitive_loss = loss_util.pixel_loss(
            self.watermark_target, extract_watermark)

        # negitive error, close to noise (all ones)
        watermark_negitive_loss = loss_util.pixel_loss(
            self.negitive_target, extract_negitive)

        watermark_total_loss = self.lambda_wm_positive * watermark_possitive_loss + \
            self.lambda_wm_negitive*watermark_negitive_loss

        # total loss, 在原先的基础上，增加水印损失
        result_set["loss_set"]["total_gen_loss"] = result_set["loss_set"]["total_gen_loss"] + \
            watermark_total_loss

    

    
