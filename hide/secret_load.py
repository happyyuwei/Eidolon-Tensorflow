import sys

import tensorflow as tf

import random

def load_image(image_file):
    image_type="jpg"
    input_num=2
    is_train=False
    image_height=256
    image_width=256

    """
    @update 2019.12.22
    新增支持多种图片格式，可选jpg,png,bmp。

    loader images, the left is grand truth, the right list is input,
    currently, one or two inputs are supported,
    use input_num to point out.
    :param image_file:
    :param image_type:
    :param input_num:
    :param is_train:
    :param image_width:
    :param image_height:
    :param flap_probability
    :param input_concat
    :param crop_width
    :param crop_height
    :return:
    """
    # 读取图片
    image = tf.io.read_file(image_file)

    # jpg格式
    if image_type == "jpg":
        image = tf.image.decode_jpeg(image)
    # png 格式
    elif image_type == "png":
        image = tf.image.decode_png(image)
    # bmp 格式
    elif image_type == "bmp":
        image = tf.image.decode_bmp(image)
    else:
        print("error: unsupported image type: {}".format(image_type))
        sys.exit()

    # only three channel currently
    # 本分软件保存的图片包含四个通道（RGBA），目前只取三个通道，去除透明度通道。
    # author yuwei
    # @since 2019.9.14
    image = image[:, :, 0:3]

    # get image width
    w = tf.shape(image)[1]

    w = w // (input_num + 1)
    # the left is label, the right is input, 1 and 2
    real_image = image[:, :w, :]
    input_image_1 = image[:, w:2 * w, :]
    if input_num == 2:
        input_image_2 = image[:, 2 * w:3 * w, :]
    else:
        # useless input 2
        input_image_2 = input_image_1

    # 从左到右， 输出， 输入
    #目前只要把输出当输入，扔掉输出
    #并且随机获取一张secret
    #------------------------------------------------------------------------

    input_image_1=real_image

    num=random.randint(1,100)

    secret_file="../../hide/img/{}.png".format(num)

    input_image_2 = tf.io.read_file(secret_file)
    input_image_2 = tf.image.decode_png(input_image_2)[:, :, 0:3]

    real_image=input_image_2
    #--------------------------------------------------------------------------

    # to float 32
    input_image_1 = tf.cast(input_image_1, tf.float32)
    input_image_2 = tf.cast(input_image_2, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    if is_train:
        # random
        # resizing to 286 x 286 x 3
        # we have to inform that the image_height is in front of  image_width,
        # because image_height is the row number of a matrix and the image_width is the column number of it.
        # The size of a matrix can be described by [row, column], so the image matrix is [image_height, image_width]
        # input_image_1 = tf.image.resize(
        #     input_image_1, [image_height + crop_height, image_width + crop_width])
        # input_image_2 = tf.image.resize(
        #     input_image_2, [image_height + crop_height, image_width + crop_width])
        # real_image = tf.image.resize(
        #     real_image, [image_height + crop_height, image_width + crop_width])

        # # 隨機裁剪
        # stacked_image = tf.stack(
        #     [input_image_1, input_image_2, real_image], axis=0)
        # cropped_image = tf.image.random_crop(
        #     stacked_image, size=[3, image_height, image_width, 3])

        # input_image_1 = cropped_image[0]
        # input_image_2 = cropped_image[1]
        # real_image = cropped_image[2]

        # if tf.random.uniform(()) < flap_probability:
        #     # random mirroring
        #     input_image_1 = tf.image.flip_left_right(input_image_1)
        #     input_image_2 = tf.image.flip_left_right(input_image_2)
        #     real_image = tf.image.flip_left_right(real_image)
        pass

    else:
        input_image_1 = tf.image.resize(
            input_image_1, size=[image_height, image_width])
        input_image_2 = tf.image.resize(
            input_image_2, size=[image_height, image_width])
        real_image = tf.image.resize(
            real_image, size=[image_height, image_width])

    # normalizing the images to [-1, 1]
    input_image_1 = (input_image_1 / 127.5) - 1
    if input_num == 2:
        input_image_2 = (input_image_2 / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    # concat if necessary
    if input_num == 2:
        input_image = tf.concat([input_image_1, input_image_2], axis=-1)
    else:
        input_image = input_image_1

    return input_image, real_image