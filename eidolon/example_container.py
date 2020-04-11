import tensorflow as tf

from eidolon import train
from eidolon.model import mnist_gan, mnist
from eidolon import loss_util
import numpy as np


# 定义一些常量
# 此处为教程直接定义，实际训练环境推荐使用配置文件定义
# 缓存大小
BUFFER_SIZE = 60000
# 批大小
BATCH_SIZE = 256
# 噪声维度
NOISE_DIM = 100


class MnistGANContainer(train.Container):
    """
    使用GAN网络生成MNIST手写体识别数字例子
    @since 2020.4.10
    @author yuwei
    """
    # 该代码已取消，继承父类时会自动调用父类构造函数
    # def __init__(self, config_loader):
    #     # 调用父类
    #     super(MnistGANContainer, self).__init__(config_loader)

    def on_prepare(self):

        # 载入数据集
        (train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
        train_images = train_images.reshape(
            train_images.shape[0], 28, 28, 1).astype('float32')
        train_images = (train_images - 127.5) / 127.5  # 将图片标准化到 [-1, 1] 区间内
        train_dataset = tf.data.Dataset.from_tensor_slices(
            train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
        # 注册数据集
        self.register_dataset(train_dataset)

        # 创建模型
        self.generator = mnist_gan.make_generator_model()
        self.discriminator = mnist_gan.make_discriminator_model()

        # 创建优化器
        generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

        # 注册模型与优化器
        self.register_model_and_optimizer(
            generator_optimizer, {"generator": self.generator}, "generator_opt")
        self.register_model_and_optimizer(discriminator_optimizer, {
                                          "discriminator": self.discriminator}, "discriminator_opt")

        # 调用父类
        super(MnistGANContainer, self).on_prepare()

    def compute_loss_function(self, each_batch, extra_batch_data):

        # 随机生成输入
        noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
        # 生成输出
        generated_images = self.generator(noise, training=True)

        # 输入真实图像后的输出
        real_output = self.discriminator(each_batch, training=True)
        # 当输入错误图像后的输出
        fake_output = self.discriminator(generated_images, training=True)

        # 计算判别损失
        gen_loss, disc_loss = loss_util.gan_loss(real_output, fake_output)

        # 返回结构集
        return {"generator_opt": gen_loss, "discriminator_opt": disc_loss}, {"generator_loss": gen_loss, "discriminator_loss": disc_loss}

    def on_test_visual(self):
        noise = tf.random.normal([4, NOISE_DIM])

        predictions = self.generator(noise, training=False)

        # 排成列表
        image_list = [predictions[0:1], predictions[1:2],
                      predictions[2:3], predictions[3:4]]
        title_list = ["1", "2", "3", "4"]
        return image_list, title_list


class MnistClassifierContainer(train.Container):
    """
    手写体识别分类训练容器教程
    @since 2020.4.11
    @author yuwei
    """

    def on_prepare(self):

        # 载入数据集
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
        train_images = train_images.reshape(
            train_images.shape[0], 28, 28, 1).astype('float32')
        train_images = (train_images - 127.5) / 127.5  # 将图片标准化到 [-1, 1] 区间内
        test_images = (test_images - 127.5) / 127.5
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (train_images, train_labels)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
        test_dataset = tf.data.Dataset.from_tensor_slices(
            (test_images, test_labels)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
        # 注册数据集
        self.register_dataset(train_dataset, test_dataset)
        # print(train_datase)

        # 创建模型
        self.model = mnist.make_DNN_model()
        # 设置优化器
        optimizer = tf.keras.optimizers.Adam(1e-4)
        # 注册模型与优化器
        self.register_model_and_optimizer(
            optimizer, {"model": self.model}, "optimizer")

        # 创建损失函数计算器
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy()

        # 注册需要记录的损失名称
        self.register_display_metrics(["train loss"])

        # 调用父类
        super(MnistClassifierContainer, self).on_prepare()

    def compute_loss_function(self, each_batch, extra_batch_data):

        # 分离输入与标签
        inputs, labels = each_batch

        outputs = self.model(inputs, training=True)

        loss = self.loss_function(labels, outputs)

        # 返回结果集
        return {"optimizer": loss}, {"train loss": loss}

    def compute_test_metrics_function(self, each_batch, extra_batch_data):

        #
        test_images, test_labels = each_batch
        predictions = self.model(test_images)

        search = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]

        test_labels=np.array(test_labels)
        labels = np.zeros([len(test_labels), 10])
        for i in range(len(test_labels)):
            labels[i]=search[test_labels[i]]


        predictions = np.array(predictions)
        predictions[predictions > 0.5] = 1
        predictions[predictions <= 0.5] = 0

        error = tf.reduce_mean(tf.abs(predictions-labels)/2)

        return {"accuracy": 1-error}

