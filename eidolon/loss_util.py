import tensorflow as tf

# 实例化二进制交叉熵损失函数计算器
binary_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def pixel_loss(gen_output, target):
    """
    像素损失，通常包括L1， L2等
    :param: gen_output 
    :param: target
    """
    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    return l1_loss


def gan_loss(disc_real_output, disc_generated_output):
    """
    GAN 网络损失，输出两个损失：gen_loss, 用于优化生成器， disc_loss, 用于优化判别器。
    gen_loss描述：生成网络是否能骗过判别器
    disc_loss描述：判别器是否能判断出生成网络的真假
    :param: disc_real_output 判决网络在输入真实图像下的输出
    :param: disc_generated_output 判决网络在输入虚假图像下的输出
    :return gen_loss, disc_loss
    """

    # 计算生成损失
    # 生成网络希望：生成的图像经过判别器，输出1， 即无法判别是假的
    gen_loss = binary_cross_entropy(tf.ones_like(
        disc_generated_output), disc_generated_output)

    # 计算判决损失
    # 判别网络希望：输入真实图片，输入1， 输入虚假图片，输出0
    real_loss = binary_cross_entropy(
        tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = binary_cross_entropy(tf.zeros_like(
        disc_generated_output), disc_generated_output)

    disc_loss = real_loss + generated_loss

    return gen_loss, disc_loss
