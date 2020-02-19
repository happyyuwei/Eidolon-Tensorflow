import tensorflow as tf

from eidolon import train_tool
from eidolon import loader
from eidolon import train
from eidolon import config
from eidolon import loss_util
from eidolon import eval_util
from eidolon.model.pixel import UNet

from hide import secret_load

import os


class SecretContainer(train.Container):

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
        # 加载数据集
        self.on_prepare_dataset()

        # 加载秘密图像数据集
        temp = self.config_loader.image_type
        self.config_loader.image_type = "png"
        secret_loader = loader.ImageLoader("../../hide/img/", is_training=True)
        self.secret_dataset = secret_loader.load(
            self.config_loader, load_function=secret_load.load_secret_image)
        self.config_loader.image_type = temp

        # 创建编码网络
        self.encoder = UNet(input_shape=[self.config_loader.image_width, self.config_loader.image_height, 6],
                            high_performance_enable=self.config_loader.high_performance)

        print("Initial encoder....")
        self.log_tool.plot_model(self.encoder, "encoder")
        print("Encoder structure plot....")

        # 创建解码网络
        self.decoder = UNet(input_shape=self.config_loader.config["dataset"]["image_size"]["value"],
                            high_performance_enable=self.config_loader.high_performance)

        print("Initial decoder....")
        self.log_tool.plot_model(self.decoder, "decoder")
        print("Decoder structure plot....")

        # 创建生成优化器
        self.optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        print("Initial optimizer....")

        # 将模型保存到存储列表，以便框架自动保存
        self.model_map["encoder"] = self.encoder
        self.model_map["decoder"] = self.decoder
        self.optimize_map["optimizer"] = self.optimizer

        # 调用父类
        super(SecretContainer, self).on_prepare()

    def compute_loss(self, input_image, target, secret_image):

        # 任务网络
        task_output = self.decoder(input_image, training=True)
        task_loss = loss_util.pixel_loss(task_output, target)

        # 输入合并
        input_tensor = tf.concat([input_image, secret_image], axis=-1)

        # 计算生成网络输出图像
        gen_output = self.encoder(input_tensor, training=True)

        # mean absolute error
        encoder_loss = loss_util.pixel_loss(gen_output, input_image)

        # 解码
        secret = self.decoder(gen_output, training=True)

        decoder_loss = loss_util.pixel_loss(secret, secret_image)

        total_loss = encoder_loss+decoder_loss+task_loss

        # 合并结果集
        loss_set = {}
        loss_set["total_loss"] = total_loss
        loss_set["encoder_loss"] = encoder_loss
        loss_set["decoder_loss"] = decoder_loss
        loss_set["task_loss"]=task_loss

        # 记录损失和输出
        result_set = {
            "loss_set": loss_set
        }

        # 返回损失
        return result_set

    def on_trainable_variables(self):
        """
        自定义训练的参数，允许重写。
        """
        return self.encoder.trainable_variables+self.decoder.trainable_variables

    # @tf.function
    def on_train_batch(self, input_image, target):
        """
        训练一批数据，该函数允许使用tensorflow加速
        """
        secret_image = None
        for each in self.secret_dataset.take(1):
            secret_image = each

        # 创建梯度计算器，负责计算损失函数当前梯度
        with tf.GradientTape() as gen_tape:

            # 计算损失
            result_set = self.compute_loss(input_image, target, secret_image)

        # 获取返回的损失
        loss_set = result_set["loss_set"]
        # 生成网络损失
        total_loss = loss_set["total_loss"]

        # 获取可训练的梯度
        generator_variables = self.on_trainable_variables()

        # 梯度求解
        generator_gradients = gen_tape.gradient(
            total_loss, generator_variables)
        # 优化网络参数
        self.optimizer.apply_gradients(
            zip(generator_gradients, generator_variables))

        # 继续返回损失，共外部工具记录
        return loss_set

    def test_metrics(self, loss_set):
        """
        定量测试，默认测试PSNR和SSIM
        """
        # 计算测试集上的PNSR与ssim
        psnr = 0
        ber = 0
        batch = 0
        for _, (test_input, test_target) in self.test_dataset.enumerate():

            for secret_image in self.secret_dataset.take(1):
                # 输入合并
                input_tensor = tf.concat([test_input, secret_image], axis=-1)
                # 生成测试结果
                predicted_image = self.encoder(input_tensor, training=True)
                result_set = eval_util.evaluate(predicted_image, test_input)
                psnr = psnr+tf.reduce_mean(result_set["psnr"])

                secret = self.decoder(predicted_image, training=True)
                result_set = eval_util.evaluate(
                    secret, secret_image, psnr_enable=False, ssim_enable=False, ber_enable=True)
                ber = ber+tf.reduce_mean(result_set["ber"])
                batch = batch+1

        psnr = psnr/batch
        ber = ber/batch

        loss_set["cover_psnr"] = psnr
        loss_set["secret_ber"] = ber

        return loss_set

    def test_visual(self):
        """
        视觉测试，在测试集上选择一个结果输出可视图像
        """
        # 测试可视化结果
        for test_input, test_target in self.test_dataset.take(1):

            for secret_image in self.secret_dataset.take(1):
                
                #task
                task_output = self.decoder(test_input, training=True)

                # 输入合并
                input_tensor = tf.concat([test_input, secret_image], axis=-1)

                # 生成测试结果
                predicted_image = self.encoder(input_tensor, training=True)

                # 解密
                secret = self.decoder(predicted_image, training=True)

        # 排成列表
        image_list = [test_input, predicted_image, secret_image, secret, task_output, test_target]
        title_list = ["C", "CS", "S_GT", "S_PR", "T_PR", "T_GT"]
        return image_list, title_list

    def on_test_epoch(self, current_epoch, loss_set):
        """
        重写测试父类方法
        """
        # 定量测试
        loss_set = self.test_metrics(loss_set)
        # 保存损失与定量测试结果
        self.log_tool.save_loss(loss_set)

        # 可视化测试
        image_list, title_list = self.test_visual()

        # 保存可视结果
        self.log_tool.save_image_list(image_list, title_list)

        # 调用父类方法
        super(SecretContainer, self).on_test_epoch(current_epoch, loss_set)
