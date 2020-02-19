import tensorflow as tf

from eidolon import train_tool
from eidolon import loader
from eidolon import train
from eidolon import config
from eidolon import loss_util
from eidolon import eval_util
from eidolon.model.pixel import UNet, Discriminator


import os


class PixelContainer(train.Container):
    """
    该类用于训练pixel-to-pixel相关的网络
    """

    def __init__(self, config_loader):
        # 调用父类
        super(PixelContainer, self).__init__(config_loader)
        # print("tensorflow version: {}".format(tf.__version__))

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
        self.log_tool.plot_model(self.generator, "generator")
        print("Generator structure plot....")

        # 创建生成优化器
        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        print("Initial generator optimizer....")

        # 将模型保存到存储列表，以便框架自动保存
        self.model_map["generator"] = self.generator
        self.optimize_map["generator_optimizer"] = self.generator_optimizer

        # 只有在用户设置使用判决网络时才会使用
        if self.config_loader.discriminator != "no":
            # 创建判决网络
            self.discriminator = Discriminator(
                input_shape=self.config_loader.config["dataset"]["image_size"]["value"])
            print("Initial discriminator....")
            self.log_tool.plot_model(self.discriminator, "discriminator")
            print("Generator discriminator plot....")

            # 创建判决优化器
            self.discriminator_optimizer = tf.keras.optimizers.Adam(
                2e-4, beta_1=0.5)
            print("Initial global discriminator optimizer....")

            # 注册模型
            self.model_map["discriminator"] = self.discriminator
            self.optimize_map["discriminator_optimizer"] = self.discriminator_optimizer

        # 调用父类
        super(PixelContainer, self).on_prepare()

    def compute_loss(self, input_image, target):
        """
        计算输出图片与目标的损失函数
        返回损失
        返回必须是一个字典，包含loss_set关键字，
        其中loss_set也是一个字典，包含total_gen_loss关键字，该关键字为总损失。
        """
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
        loss_set = {}
        loss_set["total_gen_loss"] = total_gen_loss
        # 若判决器存在，其损失才会被记录
        if self.config_loader.discriminator != "no":
            loss_set["pixel_loss"] = pixel_loss
            loss_set["gen_loss"] = gen_loss
            loss_set["disc_loss"] = disc_loss

        # 记录损失和输出
        result_set = {
            "gen_output": gen_output,
            "loss_set": loss_set
        }

        # 返回损失
        return result_set

    def on_trainable_variables(self):
        """
        自定义训练的参数，允许重写。
        """
        return self.generator.trainable_variables

        

    @tf.function
    def on_train_batch(self, each_pair, _):
        """
        训练一批数据，该函数允许使用tensorflow加速
        """
        input_image, target=each_pair
        # 创建梯度计算器，负责计算损失函数当前梯度
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

            # 计算损失
            result_set = self.compute_loss(input_image, target)

        # 获取返回的损失
        loss_set = result_set["loss_set"]
        # 生成网络损失
        total_gen_loss = loss_set["total_gen_loss"]

        #获取可训练的梯度
        generator_variables=self.on_trainable_variables()

        # 梯度求解
        generator_gradients = gen_tape.gradient(total_gen_loss, generator_variables)
        # 优化网络参数
        self.generator_optimizer.apply_gradients(zip(generator_gradients, generator_variables))
        # 只有当使用判决器时，才需要优化判决器
        if self.config_loader.discriminator != "no":
            # 判决网络损失
            disc_loss = loss_set["disc_loss"]

            discriminator_gradients = disc_tape.gradient(disc_loss,
                                                         self.discriminator.trainable_variables)

            self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                             self.discriminator.trainable_variables))
        # 继续返回损失，共外部工具记录
        return loss_set

    def test_metrics(self, loss_set):
        """
        定量测试，默认测试PSNR和SSIM
        """
        # 计算测试集上的PNSR与ssim
        psnr=0
        ssim=0
        batch=0
        for _, (test_input, test_target) in self.test_dataset.enumerate():
            # 生成测试结果
            predicted_image = self.generator(test_input, training=True)
            result_set=eval_util.evaluate(predicted_image, test_target)
            psnr=psnr+tf.reduce_mean(result_set["psnr"])
            ssim=ssim+tf.reduce_mean(result_set["ssim"])
            batch=batch+1
        
        psnr=psnr/batch
        ssim=ssim/batch

        loss_set["test_psnr"]=psnr
        loss_set["test_ssim"]=ssim

        return loss_set

    def test_visual(self):
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


    def on_test_epoch(self, current_epoch, loss_set):
        """
        重写测试父类方法
        """
        #定量测试
        loss_set=self.test_metrics(loss_set)
        # 保存损失与定量测试结果
        self.log_tool.save_loss(loss_set)

        #可视化测试
        image_list, title_list=self.test_visual()

        # 保存可视结果
        self.log_tool.save_image_list(image_list, title_list)
        
        # 调用父类方法
        super(PixelContainer, self).on_test_epoch(current_epoch, loss_set)
