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

    # 該數據第一行和第二行為總行數和列名稱，忽略。

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


def load_dataset(config_loader, is_training):
    """
    目前找不到办法兼容eidolon.load的办法，重写输入数据
    """

    # 生成目录
    if is_training == True:
        path = os.path.join(config_loader.data_dir, "train")
    else:
        path = os.path.join(config_loader.data_dir, "test")

    # 生成目录
    image_list=os.listdir(path)
    image_path_list=[]
    label_path_list=[]

    for each in image_list:
        if each.endswith(".jpg"):
            image_path_list.append(os.path.join(path, each))
            label_path_list.append(os.path.join(path, each.split(".")[0]+".txt"))
    
    #创建tf.dataset数据集
    dataset=tf.data.Dataset.from_tensor_slices((image_path_list, label_path_list))
    
    if config_loader.buffer_size <= 0:
            config_loader.buffer_size = len(image_path_list)
    if is_training == True:
        dataset = dataset.shuffle(buffer_size=config_loader.buffer_size)

    def map_function(image_file, label_file):
        # 读取图片
        image = tf.io.read_file(image_file)
        image = tf.image.decode_jpeg(image)

        #解析图片
        # 转换格式
        image = tf.cast(image, tf.float32)
        # 转成【-1,1】
        image = (image / 127.5) - 1


        #读取标签
        label=tf.io.read_file(label_file)
        #分割
        label=tf.strings.split(label, sep=" ")
        #解析
        record_defaults = list([0.0] for i in range(1)) 
        label=tf.io.decode_csv(label, record_defaults=record_defaults)
        #从【1,40】变成【40】
        label=tf.reshape(label, [40])

        return image, label

    
    dataset=dataset.map(lambda image_file, label_file: map_function(image_file, label_file))
    
    # return batch
    return dataset.batch(config_loader.batch_size)







def create_labels(img_path, label_path):
    """
    创建标签，每张图创建一个txt文件, 文件格式使用空格分开,存在为1,不存在为0.
    """

    label_map = load_labels(label_path)

    img_list = os.listdir(img_path)

    for each in img_list:

        if each.endswith(".jpg"):
            label_file = each.split(".")[0]+".txt"

            with open(os.path.join(img_path, label_file), "w") as f:
                feature = label_map[each]

                line = str(feature[0])

                for i in range(1, len(feature)):
                    line = line+" "+str(feature[i])

                f.writelines(line)
