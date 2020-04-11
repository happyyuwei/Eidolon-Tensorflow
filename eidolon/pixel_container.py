import tensorflow as tf

from eidolon import train_tool
from eidolon import loader
from eidolon import train
from eidolon import config
from eidolon import loss_util
from eidolon import eval_util
from eidolon.model.pixel import UNet, Discriminator

import numpy as np

import os


class PixelContainer(train.Container):
    """
    该类用于训练pixel-to-pixel相关的网络
    """



    def on_prepare_dataset(self):
        # 载入数据
        # 训练数据
        train_loader = loader.ImageLoader(os.path.join(
            self.config_loader.data_dir, "train"), is_training=True)
        train_dataset = train_loader.load(self.config_loader)
        # 测试数据
        test_loader = loader.ImageLoader(os.path.join(
            self.config_loader.data_dir, "test"), is_training=False)
        test_dataset = test_loader.load(self.config_loader)
        print("Load dataset, {}....".format(self.config_loader.data_dir))

        # 注册数据集
        self.register_dataset(train_dataset, test_dataset)


    def on_prepare(self):
        """
        准备阶段，完成以下事宜：
        1. 加载数据集
        2. 创建网络与优化器
        3. 将网络与优化器注册到父类中，以便自动保存
        4. 调用父类on_prepare
        """
        #加载数据集
        self.on_prepare_dataset()

        # 创建生成网络
        self.generator = UNet(input_shape=self.config_loader.config["dataset"]["image_size"]["value"],
                              high_performance_enable=self.config_loader.high_performance)

        print("Initial generator....")
        # self.log_tool.plot_model(self.generator, "generator")
        print("Generator structure plot....")

        # 创建生成优化器
        generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        print("Initial generator optimizer....")


        self.register_model_and_optimizer(generator_optimizer, {"generator":self.generator}, "generator_opt")
        print("register generator and optimizer....")



        # 只有在用户设置使用判决网络时才会使用
        if self.config_loader.discriminator != "no":
            # 创建判决网络
            self.discriminator = Discriminator(
                input_shape=self.config_loader.config["dataset"]["image_size"]["value"])
            print("Initial discriminator....")
            # self.log_tool.plot_model(self.discriminator, "discriminator")
            print("Generator discriminator plot....")

            # 创建判决优化器
            discriminator_optimizer = tf.keras.optimizers.Adam(
                2e-4, beta_1=0.5)
            print("Initial global discriminator optimizer....")

            # 注册模型
            self.register_model_and_optimizer(discriminator_optimizer, {"discriminator":self.discriminator}, "discriminator_opt")
            print("register discriminator and optimizer....")

        
        if self.config_loader.discriminator != "no":
            self.register_display_metrics(["gen_loss","disc_loss"])
        else:
            self.register_display_metrics(["gen_loss"])

        # 调用父类
        super(PixelContainer, self).on_prepare()

    def compute_loss_function(self, each_batch, extra):
        """
        本训练不需要extra参数
        计算输出图片与目标的损失函数
        返回损失
        """
        input_image, target=each_batch

        # 计算生成网络输出图像
        gen_output = self.generator(input_image, training=True)

        # mean absolute error
        pixel_loss = loss_util.pixel_loss(gen_output, target)

        if self.config_loader.discriminator != "no":

            # 输入真实的图像，计算判决网络输出
            disc_real_output = self.discriminator(
                [input_image, target], training=True)

            # 输入生成的图像，计算判决网络输出
            disc_generated_output = self.discriminator(
                [input_image, gen_output], training=True)

            # 计算GAN损失
            gen_loss, disc_loss = loss_util.gan_loss(
                disc_real_output, disc_generated_output)

            # 总的生成网络损失
            total_gen_loss = gen_loss+(100*pixel_loss)

        else:
            total_gen_loss = pixel_loss

        # 合并结果集
        loss_map = {}
        display_map={}

        loss_map["generator_opt"] = total_gen_loss
        display_map["gen_loss"]=total_gen_loss
        # 若判决器存在，其损失才会被记录
        if self.config_loader.discriminator != "no":
            loss_map["discriminator_opt"] = disc_loss
            display_map["disc_loss"]=disc_loss

        # 返回损失
        return loss_map, display_map

    
    def compute_test_metrics_function(self, each_batch, extra_batch_data):

        test_input, test_target=each_batch
        predicted_image = self.generator(test_input, training=True)
        result_set=eval_util.evaluate(predicted_image, test_target)

        return {"psnr": np.mean(result_set["psnr"]), "ssim":np.mean(result_set["ssim"])}


    def on_test_visual(self):
        """
        视觉测试，在测试集上选择一个结果输出可视图像
        """
        # 测试可视化结果
        for test_input, test_target in self.test_dataset.take(1):
            # 生成测试结果
            predicted_image = self.generator(test_input, training=True)

        # 排成列表
        image_list = [test_input, test_target, predicted_image]
        title_list = ["IN", "GT", "PR"]
        return image_list, title_list
