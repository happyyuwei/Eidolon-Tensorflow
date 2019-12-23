import tensorflow as tf



def block(filters, size, activate=tf.keras.layers.ReLU(), apply_batchnorm=True):
    """
    自动编解码器的块，每一块包括一个卷积核，输入输出尺寸不变。
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

    if activate!=None:
        result.add(activate)

    return result

def Encoder():
    """
    编码器 使用两个卷积块组成，分别为24通道和48通道。
    输入32*32*3
    输出32*32*48
    输出使用tanh激活
    """
    # 
    inputs = tf.keras.layers.Input(shape=[32, 32, 3])

    stack = [
        block(24, 3, apply_batchnorm=False),
        block(48, 3, activate=None)
    ]

    x = inputs

    # Downsampling through the model
    for down in stack:
        x = down(x)
    
    x=tf.keras.activations.tanh(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

def Decoder():
    """
    解码器 使用两个卷积块组成，分别为24通道和3通道。
    输入32*32*48
    输出32*32*3
    输出使用tanh激活
    """

    inputs = tf.keras.layers.Input(shape=[32, 32, 48])

    stack = [
        block(24, 3),
        block(3, 3, activate=None)
    ]

    x = inputs

    # Downsampling through the model
    for down in stack:
        x = down(x)
    
    x=tf.keras.activations.tanh(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


# e=Decoder()
# tf.keras.utils.plot_model(e, to_file="x.png", show_shapes=True, rankdir="TB", dpi=64)


