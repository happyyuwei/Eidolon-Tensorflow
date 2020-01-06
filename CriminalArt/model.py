import tensorflow as tf
from tensorflow.keras.applications import ResNet50

def Resnet(input_shape):
    # 创建resnet50网络
    resnet=ResNet50(weights=None, include_top=False, input_shape=input_shape)
    model=tf.keras.Sequential()
    model.add(resnet)
    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(512,activation="relu"))
    # 最后一层不是用激活函数，因为会在损失里使用softmax
    model.add(tf.keras.layers.Dense(40))

    return model

