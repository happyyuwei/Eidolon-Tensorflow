# third part lib
import tensorflow as tf
import numpy as np

# eidolon lib
from eidolon.train import Container
from eidolon import loss_util
from eidolon import loader

# inner lib
from WMNetv2 import auto

# system lib
import random
import sys


def create_mask_list():
    """
    create mask list, currently, there are four kind,left,right,up,and down crop, each crop area is 20px width
    @since 2019.9.21
    @author yuwei
    :return:
    """
    left_crop = np.ones(shape=[1, 128, 128, 3], dtype=np.float32)
    left_crop[:, :, 0:20, :] = 0

    right_crop = np.ones(shape=[1, 128, 128, 3], dtype=np.float32)
    # the last 20px, in python, we use -20
    right_crop[:, :, -20:, :] = 0

    up_crop = np.ones(shape=[1, 128, 128, 3], dtype=np.float32)
    up_crop[:, 0:20, :, :] = 0

    down_crop = np.ones(shape=[1, 128, 128, 3], dtype=np.float32)
    down_crop[:, -20:, :, :] = 0

    # add in the list
    mask_list = [left_crop, right_crop, up_crop, down_crop]
    for i in range(len(mask_list)):
        mask_list[i] = tf.convert_to_tensor(mask_list[i])
        # change size to meet the encoder output
        mask_list[i] = tf.reshape(mask_list[i], [1, 32, 32, 48])
    return mask_list


def crop_image(encoder_output, mask_list):
    """
    to train the network more robust, there are some probability to crop the output of the encoder
    the mask is 1 not crop and 0 crop.
    the encoder_output is -1 which is cropped.
    :param encoder_output:
    :param mask_list
    :return:
    """
    # there are 50% to crop the image
    if random.random() < 0.5:
        # choose a mask
        rand = random.randint(0, len(mask_list) - 1)
        mask = mask_list[rand]
        encoder_output = tf.multiply(encoder_output, mask) + mask - 1

    return encoder_output


class DecodeContainer(Container):

    def __init__(self, config_loader):
        # 调用父类
        super(DecodeContainer, self).__init__(config_loader)

    def on_prepare(self):
        """
        准备阶段，完成以下事宜：
        1. 载入数据集
        2. 创建编解码网络
        3. 创建优化器
        4. 将网络与优化器注册到父类中，以便自动保存
        5. 调用父类on_prepare
        """
        # # since the number of the ship images in cifar10-batch1 is 1025,
        # the first 1000 images are for training, then the next 25 images are for testing
        train_image_num = 1000
        test_image_num = 25
        # 载入数据集
        # load data
        if "cifar" in self.config_loader.data_dir:
            print("load cifar data....")
            self.train_dataset, self.test_dataset = loader.load_cifar(
                self.config_loader.data_dir, train_image_num, test_image_num)

        elif "mnist" in self.config_loader.data_dir:
            print("load mnist data....")
            self.train_dataset, self.test_dataset = loader.load_mnist(
                self.config_loader.data_dir, train_image_num, test_image_num)
        else:
            print("neither cifar nor mnist data found...")
            sys.exit()

        # 封装成tensorflow dataset
        self.train_dataset = tf.data.Dataset.from_tensor_slices(
            (self.train_dataset)).shuffle(train_image_num).batch(self.config_loader.batch_size)
        #测试集也打乱一下
        self.test_dataset = tf.data.Dataset.from_tensor_slices(
            (self.test_dataset)).shuffle(test_image_num).batch(self.config_loader.batch_size)

        print("create mask list....")
        self.mask_list = create_mask_list()

        # 创建编解码网络
        print("initial encoder....")
        self.encoder = auto.Encoder()
        print("initial decoder....")
        self.decoder = auto.Decoder()
        self.log_tool.plot_model(self.encoder, "encoder")
        print("Encoder structure plot....")
        self.log_tool.plot_model(self.decoder, "decoder")
        print("Decoder structure plot....")

        # initial optimizer
        print("initial optimizer....")
        self.optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        # register
        self.model_map["encoder"] = self.encoder
        self.model_map["decoder"] = self.decoder
        self.model_map["optimzer"] = self.optimizer
        # 调用父类
        super(DecodeContainer, self).on_prepare()

    @tf.function
    def train_batch(self, input_image):

         # 创建梯度计算器，负责计算损失函数当前梯度
        with tf.GradientTape() as tape:
            # encoder
            encoder_output = self.encoder(input_image, training=True)

            # crop in random, to train robust decode
            encoder_output = crop_image(encoder_output, self.mask_list)

            # decoder
            decoder_output = self.decoder(encoder_output, training=True)

            # loss
            loss = loss_util.pixel_loss(input_image, decoder_output)

        # gather variables
        # Todo i am not sure it is right to gather like this way in Tensorflow 2.0
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables

        # calculate gradients and optimize
        gradients = tape.gradient(target=loss, sources=variables)
        self.optimizer.apply_gradients(
            grads_and_vars=zip(gradients, variables))

        # 整理结果
        loss_set = {}
        loss_set["train_loss"] = loss

        return loss_set

    def on_train(self, current_epoch):
        """
        重写训练父类方法
        """
        for image_num, (input_image) in self.train_dataset.enumerate():
             # @since 2019.11.28 将该打印流变成原地刷新
            print("\r"+"input_image {}...".format(image_num), end="", flush=True)
            # 训练一个batch
            self.loss_set = self.train_batch(input_image)
        # change line
        print()
        # 调用父类方法
        super(DecodeContainer, self).on_train(current_epoch)

    def on_test(self, current_epoch):

        # save test result
        if current_epoch % self.config_loader.save_period == 0:
            # test loss on test set
            test_loss = 0
            for image_num, (input_image) in self.test_dataset.enumerate():
                encoder_output = self.encoder(input_image, training=True)
                # decoder
                decoder_output = self.decoder(encoder_output, training=True)
                # calculate loss
                test_loss = test_loss + \
                    loss_util.pixel_loss(input_image, decoder_output)

            # calculate mean loss
            test_loss = test_loss / float(image_num)

            self.loss_set["test_loss"] = test_loss
            # 保存损失
            self.log_tool.save_loss(self.loss_set)

            # 测试可视化结果
            for test_image in self.test_dataset.take(1):

                # show encoder output
                encoder_output = self.encoder(test_image, training=True)
                # crop the encoder output
                encoder_output = crop_image(encoder_output, self.mask_list)
                # decoder
                decoder_output = self.decoder(encoder_output, training=True)
                titles = ["IN", "EN", "DE"]
                image_list = [test_image, tf.reshape(
                    encoder_output, [1, 128, 128, 3]), decoder_output]
                self.log_tool.save_image_list(
                    image_list=image_list, title_list=titles)

        # 调用父类方法
        super(DecodeContainer, self).on_test(current_epoch)
