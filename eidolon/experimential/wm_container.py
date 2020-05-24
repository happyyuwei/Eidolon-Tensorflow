import tensorflow as tf

from eidolon.train import Container
from eidolon.model.pixel import UNet
from eidolon import loss_util
from eidolon import train_tool
from eidolon import loader

import numpy as np
import os

def Conv(filters, size, channel, relu=True):
    inputs=tf.keras.layers.Input(shape=[128,128,channel])
    initializer = tf.random_normal_initializer(0., 0.02)
    conv1 = tf.keras.layers.Conv2D(filters,
                                    (size, size),
                                    strides=1,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=True)
    batchnorm = tf.keras.layers.BatchNormalization()
    y = conv1(inputs)
    y = batchnorm(y)
    if relu is True:
        y = tf.nn.relu(y)
    
    return tf.keras.Model(inputs=inputs, outputs=y)

def InceptionBlock():
    # first block comes from here
    inputs=tf.keras.layers.Input(shape=[128,128,32])
    # channel 1 conv
    channel1_conv1 = Conv(32, 1, 32)
    # channel 2 conv
    channel2_conv1 = Conv(32, 1, 32)
    channel2_conv2 = Conv(32, 3, 32)
    # channel 3 conv
    channel3_conv1 = Conv(32, 1, 32)
    channel3_conv2 = Conv(32, 3, 32)
    channel3_conv3 = Conv(32, 3, 32)
    # channel combine1 conv
    combine_conv = Conv(32, 1, 96,  relu=False)

    channel1_x1 = channel1_conv1(inputs)
    # channel 2
    channel2_x1 = channel2_conv1(inputs)
    channel2_x2 = channel2_conv2(channel2_x1)
    # channel 2
    channel3_x1 = channel3_conv1(inputs)
    channel3_x2 = channel3_conv2(channel3_x1)
    channel3_x3 = channel3_conv3(channel3_x2)
    # combine
    combine_x1 = tf.concat([channel1_x1, channel2_x2, channel3_x3], axis=-1)
    combine_x2 = combine_conv(combine_x1)
    # add
    out = inputs + combine_x2
    # relu after add residual
    out = tf.nn.relu(out)


    return tf.keras.Model(inputs=inputs, outputs=out)




def ExtractInvisible(shape=[128,128,3]):

    inputs=tf.keras.layers.Input(shape=shape)

    first_conv=Conv(32, 1, shape[2])
    # block 1
    block1=InceptionBlock()
    # second block comes from here
    block2=InceptionBlock()
    # final convolution comes here
    final_conv = Conv(3, 1, 32, relu=False)

    
    out=first_conv(inputs)
    out=block1(out)
    out=block2(out)
    out=final_conv(out)

    # use tanh instead, because sigmod is [0,1] and tanh is [-1,1]
    # @since 2019.9.21
    # author yuwei
    out = tf.nn.tanh(out)
    return tf.keras.Model(inputs, out)

class WMContainer(Container):

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


        # 准备数据集
        self.on_prepare_dataset()

         # 创建生成网络
        self.generator = UNet(input_shape=self.config_loader.config["dataset"]["image_size"]["value"],
                              high_performance_enable=self.config_loader.high_performance)
        print("Initial generator....")

        self.extractor=ExtractInvisible()
        print("Initial extractor....")

        # 创建生成优化器
        optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        print("Initial optimizer....")


        self.register_model_and_optimizer(optimizer, {"generator":self.generator,"extractor":self.extractor}, "opt")
        print("register generator and optimizer....")

        self.register_display_loss(["total_loss","gen_loss", "wm_loss"])

        self.wm=train_tool.read_image("C:\\Users\\happy\\Desktop\\Evil\\Eidolon-Tensorflow\\wm.png", 128,128, change_scale=True)

        self.noise=np.zeros([1,128,128,3])

        super(WMContainer, self).on_prepare()

    def compute_loss_function(self, each_batch, extra):
        """
        本训练不需要extra参数
        计算输出图片与目标的损失函数
        返回损失
        """
        input_image, target=each_batch

        # 计算生成网络输出图像
        gen_output = self.generator(input_image, training=True)

        #提取水印
        wm_output_p=self.extractor(gen_output, training=True)

        wm_output_n=self.extractor(target, training=True)

        # mean absolute error
        pixel_loss = loss_util.pixel_loss(gen_output, target)

        wm_loss=loss_util.pixel_loss(wm_output_p, self.wm)+loss_util.pixel_loss(wm_output_n, self.noise)

        total_loss=pixel_loss+wm_loss

        return {"opt":total_loss},{"total_loss":total_loss,"gen_loss":pixel_loss, "wm_loss":wm_loss}
    
    def compute_test_metrics_function(self, each_batch, extra_batch_data):
        return {}

    def on_test_visual(self):
        """
        视觉测试，在测试集上选择一个结果输出可视图像
        """
        # 测试可视化结果
        for test_input, test_target in self.test_dataset.take(1):
            # 生成测试结果
            predicted_image = self.generator(test_input, training=True)

            predicted_wm=self.extractor(predicted_image, training=True)

        # 排成列表
        image_list = [test_input, test_target, predicted_image, predicted_wm, self.wm]
        title_list = ["IN", "GT", "PR", "wm", "WM_GT"]
        return image_list, title_list


    



