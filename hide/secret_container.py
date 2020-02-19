import tensorflow as tf

from eidolon import train_tool
from eidolon import loader
from eidolon import train
from eidolon import config
from eidolon import loss_util
from eidolon import eval_util
from eidolon.model.pixel import UNet


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
        secret_train_loader = loader.ImageLoader("../../data/QR/train", is_training=True)
        self.secret_train_dataset = secret_train_loader.load(
            image_type="png", load_function=loader.load_single_image)

         # 加载秘密图像数据集
        secret_test_loader = loader.ImageLoader("../../data/QR/test", is_training=False)
        self.secret_test_dataset = secret_test_loader.load(
            image_type="png", load_function=loader.load_single_image)

        # 创建编码网络
        self.encoder = UNet(input_shape=[self.config_loader.image_width, self.config_loader.image_height, 6],
                            high_performance_enable=self.config_loader.high_performance)

        print("Initial encoder....")
        self.log_tool.plot_model(self.encoder, "encoder")
        print("Encoder structure plot....")

        # 创建解码网络, 解码网络也是任务网络
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
        """
        param: input_image: 输入图像
        param: target: 输出图像
        param: secret_image: 秘密图像
        """

        # 训练任务网络==================================================================
        # 任务网络，常规风格转移网络任务
        task_output = self.decoder(input_image, training=True)
        # 任务网络损失，常规像素损失
        task_loss = loss_util.pixel_loss(task_output, target)

        # 功能伪装，将信息隐藏提取网络隐藏至任务网络中=====================================
        # 输入合并，当前版本只是简单把原图和秘密图像叠加在一起，一起输入至隐藏网络
        input_tensor = tf.concat([input_image, secret_image], axis=-1)

        # 秘密隐藏网络
        # 计算生成网络输出图像
        stego = self.encoder(input_tensor, training=True)

        # 信息隐藏损失：含秘图像与原图无法在视觉上体现差异
        # mean absolute error
        encoder_loss = loss_util.pixel_loss(stego, input_image)

        # 秘密提取网络
        # 提取秘密信息
        extracted_secret = self.decoder(stego, training=True)

        # 信息提取损失：提取出来的秘密信息与原信息保持一致
        decoder_loss = loss_util.pixel_loss(extracted_secret, secret_image)

        # 总损失为三者之和
        total_loss = encoder_loss+decoder_loss+task_loss

        # 合并结果集
        loss_set = {}
        loss_set["total_loss"] = total_loss
        loss_set["encoder_loss"] = encoder_loss
        loss_set["decoder_loss"] = decoder_loss
        loss_set["task_loss"] = task_loss

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

    def on_prepare_batch(self):
        """
        每次训练会随机抽取一张图像作为秘密图像
        """

        for secret_image in self.secret_train_dataset.take(1):
            pass

        # 该返回值会传递至on_train_batch（）的第二个参数（除self参数）
        return secret_image

    # 推荐添加该注解，经过测试，可节省40%时间.
    @tf.function
    def on_train_batch(self, each_pair, secret_image):
        """
        训练一批数据，该函数允许使用tensorflow加速
        """
        input_image, target = each_pair

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

        # 继续返回损失，供外部工具记录
        return loss_set

    def test_metrics(self, loss_set):
        """
        定量测试，默认测试PSNR和SSIM
        """
        # 计算测试集上的PNSR与ssim
        stego_psnr = 0
        ber = 0
        batch = 0
        for _, (test_input, test_target) in self.test_dataset.enumerate():

            # 从秘密图片中随机选择一张进行评估
            for secret_image in self.secret_test_dataset.take(1):
                # 输入合并
                input_tensor = tf.concat([test_input, secret_image], axis=-1)
                # 生成测试结果
                stego = self.encoder(input_tensor, training=True)

                # 评估stego的质量
                result_set = eval_util.evaluate(stego, test_input)
                stego_psnr = stego_psnr+tf.reduce_mean(result_set["psnr"])

                # 评估提取秘密信息的质量
                extracted_secret = self.decoder(stego, training=True)
                result_set = eval_util.evaluate(
                    extracted_secret, secret_image, psnr_enable=False, ssim_enable=False, ber_enable=True)
                ber = ber+tf.reduce_mean(result_set["ber"])
                batch = batch+1

        psnr = stego_psnr/batch
        ber = ber/batch

        loss_set["stego_psnr"] = psnr
        loss_set["secret_ber"] = ber

        return loss_set

    def test_visual(self):
        """
        视觉测试，在测试集上选择一个结果输出可视图像
        """
        # 测试可视化结果
        for test_input, test_target in self.test_dataset.take(1):

            for secret_image in self.secret_test_dataset.take(1):

                # task
                task_output = self.decoder(test_input, training=True)

                # 输入合并
                input_tensor = tf.concat([test_input, secret_image], axis=-1)

                # 生成测试结果
                predicted_image = self.encoder(input_tensor, training=True)

                # 解密
                secret = self.decoder(predicted_image, training=True)

        # 排成列表
        image_list = [test_input, predicted_image,
                      secret_image, secret, task_output, test_target]
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
