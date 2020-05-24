import tensorflow as tf



"""
mnist手写体数字识别分类网络，其训练容器见example_container.MnistClassifierContainer
@since 2019.4.11
@author yuwei
"""

def make_DNN_model(input_shape):

    #定义各层
    inputs=tf.keras.layers.Input(shape=input_shape)

    flatten=tf.keras.layers.Flatten()

    layer1=tf.keras.layers.Dense(128,activation="relu")

    dropout=tf.keras.layers.Dropout(0.2)

    layer2=tf.keras.layers.Dense(10,activation="softmax")

    #连接各层
    outputs=flatten(inputs)
    outputs=layer1(outputs)
    outputs=dropout(outputs)
    outputs=layer2(outputs)

    return tf.keras.Model(inputs=inputs,outputs=outputs)
