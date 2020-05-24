import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
import sys
import pickle

from eidolon import train_tool


def read_image(image_file, image_type, normalize=True):
    """
    载入图片，支持jpg, png, bmp
    核心调用tf.io.read_file()函数

    :param: normalize=True, 将图像转到【-1,1】区间， tf.float32
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

    if normalize == True:
        # 转换类型
        image = tf.cast(image, tf.float32)
        # 转化到【-1,1】
        image = (image / 127.5) - 1

    return image


def load_single_image(image_file, image_type, is_training, image_width, image_height, flap_probability, crop_width, crop_height):
    """
    @update 2020.4.15
    @author yuwei
    增加图像尺寸输入

    读取单一图片
    """
    # 载入
    image = read_image(image_file, image_type)

    if is_training:
        # 转换大小
        image = tf.image.resize(
            image, size=[image_height+crop_height, image_width+crop_width])
        # 随机裁剪
        image = tf.image.random_crop(
            image, size=[image_height, image_width, 3])

        # 随机翻转
        if tf.random.uniform(()) < flap_probability:
            # random mirroring
            image = tf.image.flip_left_right(image)

    else:
        # 转换大小
        image = tf.image.resize(image, size=[image_height, image_width])
        # 随机裁剪
        image = tf.image.random_crop(
            image, size=[image_height, image_width, 3])

    return image


def load_single_crop_no_rescale(image_file, image_type, is_training, image_width, image_height, flap_probability, crop_width, crop_height):
    """
    载入单张图片，不转换尺寸，仅仅在原始图片中随机裁剪指定尺寸。
    要求输入图像的尺寸大于输出尺寸，否则无法裁剪。
    """
    image = read_image(image_file, image_type)
    # 随机裁剪
    image = tf.image.random_crop(image, size=[image_height, image_width, 3])

    # 随机翻转，只在训练阶段有效
    if is_training:
        if tf.random.uniform(()) < flap_probability:
            # random mirroring
            image = tf.image.flip_left_right(image)

    return image


def load_image_default(image_file, image_type, input_num, is_train, image_width, image_height, flap_probability, crop_width, crop_height):
    """
    默认图区图片方式，为输出-输入图片对合并为一张的图片，该函数会自动拆解成两张

    @update 2019.12.22
    新增支持多种图片格式，可选jpg，png，bmp。

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
    image = read_image(image_file, image_type, normalize=False)

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


def load_label_file(path):
    """
    载入标签，返回numpy数组
    """
    return np.loadtxt(path, delimiter=",")


class ImageLoader:
    """
    由1.0版本的dataLoader修改而成。
    图像载入类，其输入输出图像需要拼接成一张图。
    左边：输出，右边：输入。若输入为多张图，则右边一次为输入1，输入2...
    """

    def __init__(self, data_dir, is_training,  file_list=None, shuffle=True):
        """
        :param data_dir: 数据集路径
        :param is_training:  是否是训练集
        :param file_list： 文件列表，若指定文件列表，则调用load()后从data_dir中载入该文件列表中的文件，忽略imgae_type类型
        :param shuffle： 打乱数据集
        """
        self.data_dir = data_dir
        self.is_training = is_training

        if file_list != None:
            for i in range(len(file_list)):
                file_list[i] = os.path.join(data_dir, file_list[i])

        self.file_list = file_list
        self.shuffle = shuffle

    def load(self, config_loader=None, load_function=None, buffer_size=0, batch_size=1, image_type="jpg", input_num=1, image_width=256, image_height=256, flap_probability=0, crop_width=0, crop_height=0, load_label_function=None):
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

        # 若输入的参数是config_loader，则先提取有效参数
        if config_loader != None:
            buffer_size = config_loader.buffer_size
            batch_size = config_loader.batch_size
            image_type = config_loader.image_type
            input_num = config_loader.input_num
            image_width = config_loader.image_width
            image_height = config_loader.image_height
            flap_probability = config_loader.data_flip
            crop_width = config_loader.crop_width
            crop_height = config_loader.crop_height

        # get dataset file list
        """
        @2019.12.26
        数据集加载无法放在GPU上，否则会报错：RuntimeError: Can't copy Tensor with type string to device /job:localhost/replica:0/task:0/device:GPU:3
        目前尚未解决该问题。
        因此指定该操作在CPU上执行
        @author yuwei
        """
        with tf.device("CPU:0"):

            if self.file_list == None:

                dir_pattern = os.path.join(
                    self.data_dir, "*.{}".format(image_type))
                """
                @since 2020.4.15 修复每次载入数据集顺序不同的bug
                @author yuwei
                需要在调用时指定shuffle=False
                """
                dataset = tf.data.Dataset.list_files(
                    dir_pattern, shuffle=False)
            else:
                dataset = tf.data.Dataset.from_tensor_slices(self.file_list)

        # recalculate buffer size
        # if buffer_size<=0， adapted the buffer size
        # then buffer size is the dataset buffer size
        # recommend buffer_size=0 to adapt the dataset size
        # @since 2019.1.18
        # @author yuwei
        # @version 0.93
        if buffer_size <= 0:
            buffer_size = len(os.listdir(self.data_dir))

        # pretreat images
        if load_function == None:
            dataset = dataset.map(lambda x: load_image_default(
                x, image_type, input_num, self.is_training, image_width, image_height, flap_probability, crop_width, crop_height))
        else:
            dataset = dataset.map(lambda x: load_function(
                x, image_type, self.is_training, image_width, image_height, flap_probability, crop_width, crop_height))

        # 如果存在标签，则载入标签数据集
        if load_label_function != None:
            # 目前默认在数据集下的labels.txt文件寻找
            labels = load_label_function(
                os.path.join(self.data_dir, "labels.txt"))
            labels = tf.cast(labels, tf.float32)
            labels = tf.data.Dataset.from_tensor_slices(labels)
            dataset = tf.data.Dataset.zip((dataset, labels))

         # shuffle the list if necessary
        if self.shuffle == True:
            dataset = dataset.shuffle(buffer_size=buffer_size)

        # return batch
        return dataset.batch(batch_size)


def process_single_image(image, is_training, *args, **kwargs):
    """
    处理单张图像

    Arguments:
        image {[图像]} -- [description]
        is_training {是否训练} -- [description]
        image_width {[图像宽度]} -- [description]
        image_height {[图像高度]} -- [description]
        image_channel {[图像通道]} -- [description]
        flap_probability {[翻转概率]} -- [description]
        crop_width {[裁剪宽度]} -- [description]
        crop_height {[裁剪高度]} -- [description]

    Returns:
      [image] -- [变换后的图像]
    """

    if len(kwargs) == 0:
        kwargs = args[0]

    # 获取参数
    image_width = kwargs["image_width"]
    image_height = kwargs["image_height"]
    image_channel = kwargs["image_channel"]
    flap_probability = kwargs["flap_probability"]
    crop_width = kwargs["crop_width"]
    crop_height = kwargs["crop_height"]

    if is_training:
        # 转换大小
        image = tf.image.resize(
            image, size=[image_height+crop_height, image_width+crop_width])
        # 随机裁剪
        image = tf.image.random_crop(
            image, size=[image_height, image_width, image_channel])

        # 随机翻转
        if tf.random.uniform(()) < flap_probability:
            # random mirroring
            image = tf.image.flip_left_right(image)

    else:
        # 转换大小，测试的时候不进行翻转与裁剪
        image = tf.image.resize(image, size=[image_height, image_width])

        # 转换类型
    image = tf.cast(image, tf.float32)
    # 转化到【-1,1】
    image = (image / 127.5) - 1

    return image


def process_image_classification(image, label, is_training,  *args, **kwargs):
    """图像分类问题预处理，将图像进行改变大小，裁剪等操作

    Arguments:
        each_pair {[each_pair]} -- [description]
        is_training {是否训练} -- [description]
        image_width {[图像宽度]} -- [description]
        image_height {[图像高度]} -- [description]
        flap_probability {[翻转概率]} -- [description]
        crop_width {[裁剪宽度]} -- [description]
        crop_height {[裁剪高度]} -- [description]

    Returns:
        [type] -- [description]
    """

    if len(kwargs) == 0:
        kwargs = args[0]

    # 获取参数
    image_width = kwargs["image_width"]
    image_height = kwargs["image_height"]
    image_channel = kwargs["image_channel"]
    flap_probability = kwargs["flap_probability"]
    crop_width = kwargs["crop_width"]
    crop_height = kwargs["crop_height"]
    label_len = kwargs["label_len"]

    # 处理图像
    image = process_single_image(image=image, is_training=is_training, image_width=image_width, image_height=image_height, image_channel=image_channel,
                                 flap_probability=flap_probability, crop_width=crop_width, crop_height=crop_height)

    # 转 one-hot 编码
    label = tf.one_hot(label, label_len)

    return image, label


def preprocess_pipline(dataset, process_function, shuffle_size, batch_size, is_training, *args):
    """[summary]
    预处理pipleline
    Arguments:
        dataset {[type]} -- [description]
        process_function {[type]} -- [description]
        shuffle_size {[type]} -- [description]
        batch_size {[type]} -- [description]
        **kwargs {[type]} -- [description]
    """
    arg = args[0]

    # 处理输入数据
    # 监督学习模式
    dataset = dataset.map((lambda x, y: process_function(
        x, y, is_training,  arg)), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # 缓存（还未清除其效果）
    dataset = dataset.cache()
    # 打乱数据
    dataset = dataset.shuffle(shuffle_size)
    # 分批
    dataset = dataset.batch(batch_size)
    # 预读取，在每次执行某数据时，后一个数据将会预读取
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


def process_images(image,target, image_type, is_training, image_width, image_height, flap_probability, crop_width, crop_height):
    """


    @update 2019.12.22
    新增支持多种图片格式，可选jpg，png，bmp。

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
    image = read_image(image, image_type[0], normalize=False)

    target = read_image(target, image_type[1], normalize=False)

    if is_training:
        # random
        # resizing to 286 x 286 x 3
        # we have to inform that the image_height is in front of  image_width,
        # because image_height is the row number of a matrix and the image_width is the column number of it.
        # The size of a matrix can be described by [row, column], so the image matrix is [image_height, image_width]
        image = tf.image.resize(
            image, [image_height + crop_height, image_width + crop_width])
        target = tf.image.resize(
            target, [image_height + crop_height, image_width + crop_width])

        # 隨機裁剪，统一裁剪
        stacked_image = tf.stack([image, target], axis=0)
        cropped_image = tf.image.random_crop(
            stacked_image, size=[2, image_height, image_width, 3])

        image = cropped_image[0]
        target = cropped_image[1]

        # 统一翻转
        if tf.random.uniform(()) < flap_probability:
            # random mirroring
            image = tf.image.flip_left_right(image)
            target = tf.image.flip_left_right(target)

    else:
        image = tf.image.resize(
            image, [image_height + crop_height, image_width + crop_width])
        target = tf.image.resize(
            target, [image_height + crop_height, image_width + crop_width])

        # to float 32
    image = tf.cast(image, tf.float32)
    target = tf.cast(target, tf.float32)

    # normalizing the images to [-1, 1]
    image = (image / 127.5) - 1
    target = (target / 127.5) - 1

    return image, target


def load_custom_image_dataset(config_loader, path="", features=[], data_types=[],  batch_size=1, is_training=True, image_width=128, image_height=128, flap_probability=0, crop_width=0, crop_height=0):
    """
    @since 2020.4.15 修复每次载入数据集顺序不同的bug
    @author yuwei
    需要在调用时指定shuffle=False
    """

    #若载入配置文件，则后续参数输入无效
    if config_loader != None:
        features=config_loader.data_features
        data_types=config_loader.data_types
        batch_size = config_loader.batch_size
        image_width = config_loader.image_width
        image_height = config_loader.image_height
        flap_probability = config_loader.data_flip
        crop_width = config_loader.crop_width
        crop_height = config_loader.crop_height

        if is_training==True:
            path=os.path.join(config_loader.data_dir, "train")
        else:
            path=os.path.join(config_loader.data_dir, "test")


    dataset_list = []
    shuffle_size = -1

    for i in range(len(features)):

        data_dir = os.path.join(path, features[i])
        shuffle_size = len(os.listdir(data_dir))
        dir_pattern = os.path.join(data_dir, "*.{}".format(data_types[i]))
        dataset = tf.data.Dataset.list_files(dir_pattern, shuffle=False)
        dataset_list.append(dataset)

    # 建立数据集映射关系
    dataset = tf.data.Dataset.zip(tuple(dataset_list))
    dataset = dataset.map((lambda x,y: process_images(x,y, data_types , is_training, image_width,
                                                    image_height, flap_probability, crop_width, crop_height)), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # 缓存（还未清除其效果）
    dataset = dataset.cache()
    # 打乱数据
    dataset = dataset.shuffle(shuffle_size)
    # 分批
    dataset = dataset.batch(batch_size)
    # 预读取，在每次执行某数据时，后一个数据将会预读取
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


def load_dataset(preprocess_function, batch_size,  data_dir=None, tfds_name=None, **kwargs):
    """
    载入数据集
    @since 2020.5.15

    Arguments:
        mode {[str]} -- [载入模式，可选：supervise，unsupervise。当数据集包括输入和标签，使用supervise]
        preprocess_function {[数据预处理函数]} -- [description]
        batch_size {[批处理大小]} -- [description]

    Keyword Arguments:
        data_dir {[数据位置]} -- [载入位置] (default: {None})
        tfds_name {[载入tfds数据集]} -- [description] (default: {None})

    Returns:
        [type] -- [description]
    """

    if tfds_name != None:
        # 调用tfds接口加载数据，将数据集划分成train和test
        # as_supervided=True，返回输入输出标签对，否则返回所有字典数据，with_info包含数据集信息
        (train_dataset, test_dataset), info = tfds.load(name=tfds_name, data_dir=data_dir,
                                                        split=["train", "test"], as_supervised=True,  with_info=True)
        # shuffle size
        train_shuffle_size = info.splits['train'].num_examples
        test_shuffle_size = info.splits['test'].num_examples
        # 处理 Pipleline
        train_dataset = preprocess_pipline(
            train_dataset, preprocess_function, train_shuffle_size, batch_size, True, kwargs)
        test_dataset = preprocess_pipline(
            test_dataset, preprocess_function, test_shuffle_size, batch_size, False, kwargs)

        return train_dataset, test_dataset


def load_image_dataset(config_loader=None, preprocess_function=None, batch_size=64,  data_dir=None, tfds_name=None, image_width=28, image_height=28, image_channel=1, crop_width=0, crop_height=0, flap_probability=0, label_len=0):
    """
    载入图像数据集，接收config_loader配置或者参数配置
    如果两者均制定，则以config_loader为准

    Keyword Arguments:
        config_loader {[type]} -- [description] (default: {None})
        preprocess_function {[type]} -- [description] (default: {None})
        batch_size {int} -- [description] (default: {64})
        data_dir {[type]} -- [description] (default: {None})
        tfds_name {[type]} -- [description] (default: {None})
        is_training {bool} -- [description] (default: {True})
        image_width {int} -- [description] (default: {28})
        image_height {int} -- [description] (default: {28})
        image_channel {int} -- [description] (default: {1})
        crop_width {int} -- [description] (default: {0})
        crop_height {int} -- [description] (default: {0})
        flap_probability {int} -- [description] (default: {0})
    """
    if config_loader != None:
        preprocess_function = config_loader.preprocess_function
        batch_size = config_loader.batch_size
        data_dir = config_loader.data_dir
        tfds_name = config_loader.tf_data
        image_width = config_loader.image_width
        image_height = config_loader.image_height
        image_channel = config_loader.image_channel
        crop_width = config_loader.crop_width
        crop_height = config_loader.crop_height
        flap_probability = config_loader.data_flip
        label_len = len(config_loader.classify_labels)

     # 载入预处理函数
    # 解析模型
    if str(type(preprocess_function)) == "<class 'str'>":
        preprocess_function = train_tool.load_function(
            config_loader.preprocess_function)

    # 调用 核心函数
    train_dataset, test_dataset = load_dataset(preprocess_function=preprocess_function, batch_size=batch_size,  data_dir=data_dir, tfds_name=tfds_name, image_width=image_width,
                                               image_height=image_height, image_channel=image_channel, crop_width=crop_width, crop_height=crop_height, flap_probability=flap_probability, label_len=label_len)
    return train_dataset, test_dataset
