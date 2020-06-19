import tensorflow as tf

from eidolon import train, loader
from eidolon import loss_util, train_tool
import numpy as np


import os
import sys
import importlib


class ClassifyContainer(train.Container):

    def on_prepare(self):

        # 载入数据集
        train_dataset, test_dataset = loader.load_image_dataset(
            config_loader=self.config_loader)
        self.register_dataset(train_dataset, test_dataset)

        # 创建模型工厂
        model_factory = train_tool.load_function(self.config_loader.classify_model_function)
        # 创建模型
        self.model = model_factory(self.config_loader.image_size)

        # 设置优化器
        optimizer = tf.keras.optimizers.Adam(1e-4)

        # 注册模型与优化器
        self.register_model_and_optimizer(
            optimizer, {"model": self.model}, "optimizer")
        print("Initial model and optimizer....")

        # 注册需要记录的损失名称
        self.register_display_loss(["loss"])

        #创建损失函数对象
        self.loss_function=train_tool.load_function(self.config_loader.classify_loss_function)()

        # 调用父类
        super(ClassifyContainer, self).on_prepare()

    def compute_loss_function(self, each_batch, extra_batch_data):
        """
        损失函数
        Returns:
            [type] -- [description]
        """

        # 分离输入与标签
        inputs, labels = each_batch

        # 计算输出
        outputs = self.model(inputs, training=True)

        # 计算损失函数
        loss = self.loss_function(labels, outputs)

        # 返回结果集
        return {"optimizer": loss}, {"loss": loss}

    def visual_function(self, each_batch, extra_batch_data):
        """
        视觉测试，在测试集上选择一个结果输出可视图像
        """
        test_input, test_label=each_batch

        # 结果图像
        image_list = []
        # 标题
        title_list = []

        # 生成测试结果
        predicted_label = self.model(test_input)

        # result = train_tool.visual_classify(
        #     test_input[0], labels, np.array(predicted_label[0]))

        #添加
        image_list.append(test_input)
        title_list.append("img")

        # predicted_label=np.array()

        # #计算预测出来的名字
        # for i in range()
        # predict_name=self.config_loader.classify_labels

        return {
            "image": {
                "image_list": image_list,
                "title_list": title_list,
            },
            "description":{
                 "predict_label": predicted_label,
                 "target_label": test_label
            }
        }

    def compute_test_metrics_function(self, each_batch, extra_batch_data):


        test_images, test_labels = each_batch
        predictions = self.model(test_images)

        test_labels = np.array(test_labels)

        predictions = np.array(predictions)
        predictions[predictions > 0.5] = 1
        predictions[predictions <= 0.5] = 0

        error = tf.reduce_mean(tf.abs(predictions-test_labels)/2)

        return {"accuracy": 1-error}
