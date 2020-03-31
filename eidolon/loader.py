import tensorflow as tf
import numpy as np
import os
import sys
import pickle


def read_image(image_file, image_type):
    """
    载入图片，支持jpg, png, bmp
    核心调用tf.io.read_file()函数
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
    return image


def load_single_image(image_file):
    """
    读取单一图片
    """
    # 载入
    image = read_image(image_file, "png")
    # 转换类型
    image = tf.cast(image, tf.float32)
    # 转换大小
    image = tf.image.resize(image, size=[256, 256])
    # 转化到【-1,1】
    image = (image / 127.5) - 1

    return image


def load_image_default(image_file, image_type, input_num, is_train, image_width, image_height, flap_probability, crop_width, crop_height):
    """
    默认图区图片方式，为输出-输入图片对合并为一张的图片，该函数会自动拆解成两张

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
    image = read_image(image_file, image_type)

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
        input_image_1 = tf.image.resize(
            input_image_1, [image_height + crop_height, image_width + crop_width])
        input_image_2 = tf.image.resize(
            input_image_2, [image_height + crop_height, image_width + crop_width])
        real_image = tf.image.resize(
            real_image, [image_height + crop_height, image_width + crop_width])

        # 隨機裁剪
        stacked_image = tf.stack(
            [input_image_1, input_image_2, real_image], axis=0)
        cropped_image = tf.image.random_crop(
            stacked_image, size=[3, image_height, image_width, 3])

        input_image_1 = cropped_image[0]
        input_image_2 = cropped_image[1]
        real_image = cropped_image[2]

        if tf.random.uniform(()) < flap_probability:
            # random mirroring
            input_image_1 = tf.image.flip_left_right(input_image_1)
            input_image_2 = tf.image.flip_left_right(input_image_2)
            real_image = tf.image.flip_left_right(real_image)

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


class ImageLoader:
    """
    由1.0版本的dataLoader修改而成。
    图像载入类，其输入输出图像需要拼接成一张图。
    左边：输出，右边：输入。若输入为多张图，则右边一次为输入1，输入2...
    """

    def __init__(self, data_dir, is_training,  file_list=None):
        """
        :param train_dir:
        :param is_training:
        :param file_list
        """
        self.data_dir = data_dir
        self.is_training = is_training

        if file_list!=None:
            for i in range(len(file_list)):
                file_list[i]=os.path.join(data_dir,file_list[i])

        self.file_list=file_list

    def load(self, config_loader=None, load_function=None, buffer_size=0, batch_size=1, image_type="jpg", input_num=1, is_train=True, image_width=256, image_height=256, flap_probability=0, crop_width=0, crop_height=0):
        """
        @update 2019.2.19
        去除对config_loader配置类的依赖

        允许使用config_loader作为参数输入， 也可以直接传递参数作为输入。
        如果config_loader不为None，则后面的参数默认无效。
        """
        # 判断数据存在
        if os.path.exists(self.data_dir) == False:
            print("Error: Dataset not exist. {}".format(self.data_dir))
            sys.exit()

        #若输入的参数是config_loader，则先提取有效参数
        if config_loader!=None:
            buffer_size=config_loader.buffer_size
            batch_size=config_loader.batch_size
            image_type=config_loader.image_type
            input_num=config_loader.input_num
            image_width=config_loader.image_width
            image_height=config_loader.image_height
            flap_probability=config_loader.data_flip
            crop_width=config_loader.crop_width
            crop_height=config_loader.crop_height

        # get dataset file list
        """
        @2019.12.26 
        数据集加载无法放在GPU上，否则会报错：RuntimeError: Can't copy Tensor with type string to device /job:localhost/replica:0/task:0/device:GPU:3
        目前尚未解决该问题。
        因此指定该操作在CPU上执行
        @author yuwei
        """
        with tf.device("CPU:0"):

            if self.file_list==None:
                dataset = tf.data.Dataset.list_files(
                    os.path.join(self.data_dir, "*.{}".format(image_type)))
            else:
                dataset=tf.data.Dataset.from_tensor_slices(self.file_list)

        # recalculate buffer size
        # if buffer_size<=0， adapted the buffer size
        # then buffer size is the dataset buffer size
        # recommend buffer_size=0 to adapt the dataset size
        # @since 2019.1.18
        # @author yuwei
        # @version 0.93
        if buffer_size <= 0:
            buffer_size = len(os.listdir(self.data_dir))

        # shuffle the list if necessary
        if self.is_training == True:
            dataset = dataset.shuffle(buffer_size=buffer_size)

        # pretreat images
        if load_function == None:
            dataset = dataset.map(lambda x: load_image_default(
                x, image_type, input_num, self.is_training, image_width, image_height, flap_probability, crop_width, crop_height))
        else:
            dataset = dataset.map(lambda x: load_function(x))

        # return batch
        return dataset.batch(batch_size)


"""
The functions below may be deleted in the future.
@since 2020.2.17
@author yuwei
"""


def load_cifar(cifar_file, train_num, test_num, image_shape=[32, 32]):
    """
    this function is to use to load cifar images to tensor, scale=[-1,1]
    @update 2019.12.24
    允许修改图像的尺寸,输入的尺寸为二维宽高，通道数固定为3

    @update 2019.11.30
    移至loader文件作为基础设施
    @since 2019.9.20
    @author yuwei
    :param cifar_file:
    :param train_num:
    :param test_num:
    :param image_shape:
    :return:
    """
    with open(cifar_file, 'rb') as fo:
        # load dictionary
        dict = pickle.load(fo, encoding='bytes')
        # load labels
        labels = dict[b'labels']
        # load images binary
        imgs = dict[b'data']

    # image list
    image_list = []
    # currently, I only use ship images, the label is 8
    for i in range(len(labels)):
        if labels[i] == 8:
            # reshape the image to RGB channel
            img = np.reshape(imgs[i], (3, 32, 32))
            temp = np.zeros([32, 32, 3])
            temp[:, :, 0] = img[0, :, :]
            temp[:, :, 1] = img[1, :, :]
            temp[:, :, 2] = img[2, :, :]
            img = temp
            image_list.append(img)

    image_array = np.zeros([len(image_list), 32, 32, 3])
    # to image array, the first axis is image number, then the three change is RGB
    for i in range(len(image_array)):
        image_array[i, :, :, :] = image_list[i]

    # to float32 tensor
    image_array = tf.convert_to_tensor(image_array.astype("float32"))
    # 修改尺寸
    image_array = tf.image.resize(image_array, image_shape)

    # 变成【-1,1】
    image_array = (image_array - 127.5) / 127.5

    return image_array[0:train_num, :, :, :], image_array[train_num:train_num + test_num, :, :, :]


def load_mnist(mnist_file, train_num, test_num, image_shape=[32, 32]):
    """
    load the mnist dataset and change it to binary images，, scale=[-1,1]

    @update 2019.12.24
    允许修改图像的尺寸,输入的尺寸为二维宽高，通道数固定为3

    @update 2019.12.3
    移至loader文件作为基础设施
    :param mnist_file:
    :param train_num:
    :param test_num:
    :return:
    """
    # load file，the file [0,255]
    images = np.load(mnist_file)
    # change shape,数据集中mnist尺寸28*28*1
    images = images.reshape(images.shape[0], 28, 28, 1).astype("float32")
    images = tf.image.resize(images, image_shape)
    images = (images - 127.5) / 127.5  # Normalize the images to [-1, 1]
    images = np.array(images)
    # binary value -1 and 1
    images[images >= 0] = 1
    images[images < 0] = -1

    # 变成三通道
    temp = np.zeros([np.shape(images)[0], image_shape[0], image_shape[1], 3])
    temp[:, :, :, 0] = images[:, :, :, 0]
    temp[:, :, :, 1] = images[:, :, :, 0]
    temp[:, :, :, 2] = images[:, :, :, 0]

    # convert to tensor
    images = tf.convert_to_tensor(temp.astype("float32"))
    # the first is train images and the second is test images
    return images[0:train_num, :, :, :], images[train_num:train_num + test_num, :, :, :]
