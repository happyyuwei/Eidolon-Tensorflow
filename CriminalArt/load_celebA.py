"""
载入celebA数据集的图片以及标签，标签为40维的特征
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import os

from eidolon import loader


def load_labels(path, one_hot=False):

    with open(path, "r") as f:
        lines = f.readlines()

    
    #該數據第一行和第二行為總行數和列名稱，忽略。

    lines = lines[2:]
    lines = [each.strip() for each in lines]
    list_map = {}
    for each in lines:
        each = each.replace("  ", " ")
        arr = each.split(" ")

        value = arr[1:]

        value = [float(e) for e in value]

        if one_hot == True:
            # 特征为40维度,每个特征用2维表示，第一维为1，则为正，否则为假
            feature = np.zeros([len(value), 2])
            for i in range(len(value)):
                if value == 1:
                    feature[i, :] = [1, 0]
                else:
                    feature[i, :] = [0, 1]

        else:
            feature = (np.array(value)+1)/2

        list_map[arr[0]] = feature

    return list_map



# load_labels("C:\\Users\\happy\\Downloads\\list_attr_celeba.txt")


def load_dataset(config_loader, is_training):
    """
    目前找不到办法兼容eidolon.load的办法，重写输入数据
    """

    # labels在celebA的根目录下， 名称为list_attr_celeba.txt
    label_map = load_labels(os.path.join(
        config_loader.data_dir, "list_attr_celeba.txt"))

    # 生成目录
    if is_training == True:
        path = os.path.join(config_loader.data_dir, "train")
    else:
        path = os.path.join(config_loader.data_dir, "test")

    # 图片列表
    image_list = os.listdir(path)

    image_num = len(image_list)

    # 图片结果集
    images = np.zeros([image_num, config_loader.image_height, config_loader.image_width, 3])
    #标签结果集, 每个标签40维
    labels = np.zeros([image_num, 40])

    # 加载图片
    for i in range(image_num):

         # 存储
        images[i] = plt.imread(os.path.join(path, image_list[i]))[:, :, 0:3]
        # 查询标签
        labels[i] = label_map[image_list[i]]


     # 转换格式
    images = tf.cast(images, tf.float32)
    # 转成【-1,1】
    images = (images / 127.5) - 1

    #將標籤轉換成float32
    labels=tf.cast(labels, tf.float32)


        # 封装成tensorflow.dataset
    return tf.data.Dataset.from_tensor_slices(
        (images, labels)).shuffle(image_num).batch(config_loader.batch_size)
