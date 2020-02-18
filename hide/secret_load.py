import sys
import tensorflow as tf
import random

from eidolon import loader

def load_secret_image(image_file):
    #载入
    image = loader.load(image_file, "png")
    #转换类型
    image = tf.cast(image, tf.float32)
    #转换大小
    image = tf.image.resize(image, size=[256, 256])
    #转化到【-1,1】
    image = (image / 127.5) - 1

    return image
