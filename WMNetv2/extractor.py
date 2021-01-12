import tensorflow as tf


def Conv(filters, size, activate=tf.keras.layers.ReLU(), apply_batchnorm=True):
    """
    提取器的卷积块，每一块包括一个卷积核，输入输出尺寸不变。
    可选择使用relu激活
    使用偏置以及batchnorm。
    """
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=1, padding='same',
                               kernel_initializer=initializer, use_bias=True))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    if activate != None:
        result.add(activate)

    return result


def Extractor(input_shape):
    """
    水印提取器，输入输出尺寸一样
    """
    inputs = tf.keras.layers.Input(shape=input_shape)

    first_conv = Conv(32, 1)

    # first block comes from here
    # channel 1 conv
    block1_channel1_conv1 = Conv(32, 1)
    # channel 2 conv
    block1_channel2_conv1 = Conv(32, 1)
    block1_channel2_conv2 = Conv(32, 3)
    # channel 3 conv
    block1_channel3_conv1 = Conv(32, 1)
    block1_channel3_conv2 = Conv(32, 3)
    block1_channel3_conv3 = Conv(32, 3)
    # channel combine1 conv
    block1_combine_conv = Conv(32, 1, activate=None)

    # second block comes from here
    # channel 1 conv
    block2_channel1_conv1 = Conv(32, 1)
    # channel 2 conv
    block2_channel2_conv1 = Conv(32, 1)
    block2_channel2_conv2 = Conv(32, 3)
    # channel 3 conv
    block2_channel3_conv1 = Conv(32, 1)
    block2_channel3_conv2 = Conv(32, 3)
    block2_channel3_conv3 = Conv(32, 3)
    # channel combine1 conv
    block2_combine_conv = Conv(32, 1, activate=None)

    # final convolution comes here
    final_conv = Conv(3, 1, activate=None)

    x = inputs
    # 连接开始
    # extend channel
    x = first_conv(x)
    # block 1 comes from here
    # channel 1
    block1_channel1_x1 = block1_channel1_conv1(x)
    # channel 2
    block1_channel2_x1 = block1_channel2_conv1(x)
    block1_channel2_x2 = block1_channel2_conv2(block1_channel2_x1)
    # channel 2
    block1_channel3_x1 = block1_channel3_conv1(x)
    block1_channel3_x2 = block1_channel3_conv2(block1_channel3_x1)
    block1_channel3_x3 = block1_channel3_conv3(block1_channel3_x2)
    # combine
    block1_combine_x1 = tf.keras.layers.concatenate(
        [block1_channel1_x1, block1_channel2_x2, block1_channel3_x3])
    block1_combine_x2 = block1_combine_conv(block1_combine_x1)
    # add
    block1_out = x + block1_combine_x2
    # relu after add residual
    block1_out = tf.nn.relu(block1_out)

    # block 2 comes from here
    # channel 1
    block2_channel1_x1 = block2_channel1_conv1(block1_out)
    # channel 2
    block2_channel2_x1 = block2_channel2_conv1(block1_out)
    block2_channel2_x2 = block2_channel2_conv2(block2_channel2_x1)
    # channel 2
    block2_channel3_x1 = block2_channel3_conv1(block1_out)
    block2_channel3_x2 = block2_channel3_conv2(block2_channel3_x1)
    block2_channel3_x3 = block2_channel3_conv3(block2_channel3_x2)
    # combine
    block2_combine_x1 = tf.keras.layers.concatenate(
        [block2_channel1_x1, block2_channel2_x2, block2_channel3_x3], axis=-1)
    block2_combine_x2 = block2_combine_conv(block2_combine_x1)
    # add
    block2_out = block1_out + block2_combine_x2
    block2_out = tf.nn.relu(block2_out)

    # final convolution comes from here
    out = final_conv(block2_out)

    # use tanh instead, because sigmod is [0,1] and tanh is [-1,1]
    # @since 2019.9.21
    # author anomymity
    out = tf.nn.tanh(out)

    return tf.keras.Model(inputs=inputs, outputs=out)


# e=Extractor([256,256,3])
# tf.keras.utils.plot_model(e, show_shapes=True, dpi=64)
