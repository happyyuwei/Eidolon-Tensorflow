"""
本脚本用于预处理数据集
"""
import tensorflow as tf
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import os
from eidolon.loader import read_image
# --------------------------------------------------------------------------------
# Definitions
# --------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple('Label', [

    'name',  # The identifier of this label, e.g. 'car', 'person', ... .
    # We use them to uniquely name a class

    'id',  # An integer ID that is associated with this label.
    # The IDs are used to represent the label in ground truth images
    # An ID of -1 means that this label does not have an ID and thus
    # is ignored when creating ground truth images (e.g. license plate).
    # Do not modify these IDs, since exactly these IDs are expected by the
    # evaluation server.

    'trainId',  # Feel free to modify these IDs as suitable for your method. Then create
    # ground truth images with train IDs, using the tools provided in the
    # 'preparation' folder. However, make sure to validate or submit results
    # to our evaluation server using the regular IDs above!
    # For trainIds, multiple labels might have the same ID. Then, these labels
    # are mapped to the same class in the ground truth images. For the inverse
    # mapping, we use the label that is defined first in the list below.
    # For example, mapping all void-type classes to the same ID in training,
    # might make sense for some approaches.
    # Max value is 255!

    'category',  # The name of the category that this label belongs to

    'categoryId',  # The ID of this category. Used to create ground truth images
    # on category level.

    'hasInstances',  # Whether this label distinguishes between single instances or not

    'ignoreInEval',  # Whether pixels having this class as ground truth label are ignored
    # during evaluations or not

    'color',  # The color of this label
])


# --------------------------------------------------------------------------------
# A list of all labels
# --------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for your approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label('unlabeled',  0,      255, 'void', 0, False, True, (0,  0,  0)),
    Label('dynamic',  1,      255, 'void', 0, False, True, (111, 74,  0)),
    Label('ground',  2,      255, 'void', 0, False, True, (81,  0, 81)),
    Label('road',  3,        0, 'flat', 1, False, False, (128, 64, 128)),
    Label('sidewalk',  4,        1, 'flat', 1, False, False, (244, 35, 232)),
    Label('parking',  5,      255, 'flat', 1, False, True, (250, 170, 160)),
    Label('rail track',  6,      255, 'flat', 1, False, True, (230, 150, 140)),
    Label('building',  7,        2, 'construction',
          2, False, False, (70, 70, 70)),
    Label('wall',  8,        3, 'construction',
          2, False, False, (102, 102, 156)),
    Label('fence',  9,        4, 'construction',
          2, False, False, (190, 153, 153)),
    Label('guard rail', 10,      255, 'construction',
          2, False, True, (180, 165, 180)),
    Label('bridge', 11,      255, 'construction',
          2, False, True, (150, 100, 100)),
    Label('tunnel', 12,      255, 'construction',
          2, False, True, (150, 120, 90)),
    Label('pole', 13,        5, 'object', 3, False, False, (153, 153, 153)),
    Label('polegroup', 14,      255, 'object',
          3, False, True, (153, 153, 153)),
    Label('traffic light', 15,        6, 'object',
          3, False, False, (250, 170, 30)),
    Label('traffic sign', 16,        7, 'object',
          3, False, False, (220, 220,  0)),
    Label('vegetation', 17,        8, 'nature',
          4, False, False, (107, 142, 35)),
    Label('terrain', 18,        9, 'nature', 4, False, False, (152, 251, 152)),
    Label('sky', 19,       10, 'sky', 5, False, False, (70, 130, 180)),
    Label('person', 20,       11, 'human', 6, True, False, (220, 20, 60)),
    Label('rider', 21,       12, 'human', 6, True, False, (255,  0,  0)),
    Label('car', 22,       13, 'vehicle', 7, True, False, (0,  0, 142)),
    Label('truck', 23,       14, 'vehicle', 7, True, False, (0,  0, 70)),
    Label('bus', 24,       15, 'vehicle', 7, True, False, (0, 60, 100)),
    Label('caravan', 25,      255, 'vehicle', 7, True, True, (0,  0, 90)),
    Label('trailer', 26,      255, 'vehicle', 7, True, True, (0,  0, 110)),
    Label('train', 27,       16, 'vehicle', 7, True, False, (0, 80, 100)),
    Label('motorcycle', 28,       17, 'vehicle', 7, True, False, (0,  0, 230)),
    Label('bicycle', 29,       18, 'vehicle', 7, True, False, (119, 11, 32)),
    Label('license plate', 30,       -1,
          'vehicle', 7, False, True, (0,  0, 142)),
]


def get_label_id(color):
    """[summary]
    将颜色映射到标签ID
    Arguments:
        color {[type]} -- [description]
    """
    # 遍历所有颜色
    min_distance = 99999999
    index = -1
    for i in range(len(labels)):
        diff = np.array(labels[i].color)-color
        distance = np.sum(diff*diff)

        if min_distance > distance:
            min_distance = distance
            index = i

    return index


def seg_to_label(img):
    """[summary]
    将输入的分割图像转换成标签，每个像素使用一个整数标签。标签参照上标。
    Arguments:
        img {[numpy 矩阵]} -- [输入为三维矩阵，图像为三通道]
    """

    height, width, _ = np.shape(img)

    label_map = np.zeros([height, width])

    for i in range(height):
        for j in range(width):
            label_map[i, j] = get_label_id(img[i, j, :])

    return label_map


def segimage_to_label(img_path, map_path):
    """[summary]
    分割图转换成标签图
    Arguments:
        img_path {[type]} -- [description]
    """
    # 读取图片
    img = plt.imread(img_path)[:, :, 0:3]
    # img=img*255
    # 转成标签
    label_map = seg_to_label(img)
    # 保存成文本
    np.savetxt(map_path, label_map, fmt="%d")
    


def label_to_segimage(label_path):
    """[summary]
    将标签文件转化成分割图像
    Arguments:
        label_path {[type]} -- [description]
    """
    # 载入txt
    label_map = np.loadtxt(label_path, dtype=np.uint8)
    height, width = np.shape(label_map)

    # 创建图像模板
    seg_img = np.zeros([height, width, 3], dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            seg_img[i, j, :] = np.array(labels[label_map[i, j]].color)

    return seg_img

def script(path):
    """批量生成脚本，仅在准备数据集的时候使用

    Arguments:
        path {[type]} -- [description]
    """
    seg_path = os.path.join(path, "seg")
    os.mkdir(os.path.join(path, "label"))
    img_list = os.listdir(os.path.join(path, "image"))

    for i in range(len(img_list)):
        segimage_to_label(os.path.join(seg_path, img_list[i]), os.path.join(
            path, "label", img_list[i].replace("jpg", "txt")))

        print(i)


# def script_ex(path):
#     """
#     将以txt保存的label转换成png保存的label.
#     txt的label太大，且难以适用tf读取

#     Arguments:
#         path {[type]} -- [description]
#     """
#     # 载入txt
#     label_map = np.loadtxt(path, dtype=np.uint8)
#     plt.imsave("1.png", label_map)




def preprocess(image_file, label_file):
    print(label_file)

    #读取文件
    image = read_image(image_file, "jpg", normalize=False)
    #读取标签
    # label_map = tf.io.decode_csv()

    return image, label_map


def load_dataset(path, batch_size):
    """
    加载指定路径下的数据集，返回tf.data.dataset结构

    Arguments:
        path {[type]} -- [description]
    """

    #图像数据集
    image_dir = os.path.join(path, "image")
    image_dir_pattern = os.path.join(image_dir, "*.jpg")
    image_dataset = tf.data.Dataset.list_files(image_dir_pattern, shuffle=False)

    #标签数据集
    label_dir = os.path.join(path, "label")
    label_dir_pattern = os.path.join(label_dir, "*.txt")
    label_dataset = tf.data.Dataset.list_files(label_dir_pattern, shuffle=False)

    #数据集大小
    shuffle_size = len(os.listdir(image_dir))

    #建立映射关系
    dataset = tf.data.Dataset.zip((image_dataset, label_dataset))
    #数据预处理
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # 缓存（还未清除其效果）
    dataset = dataset.cache()
    # 打乱数据
    dataset = dataset.shuffle(shuffle_size)
    # 分批
    dataset = dataset.batch(batch_size)
    # 预读取，在每次执行某数据时，后一个数据将会预读取
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

# print(os.listdir("../../../data/city/train"))