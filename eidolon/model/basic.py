import tensorflow as tf

"""
包括3层卷积与池化的基本网络
"""
def make_Conv_models(input_shape):

    inputs=tf.keras.layers.Input(shape=input_shape)

    conv1=tf.keras.layers.Conv2D(16,3, padding="same", activation="relu")
    pooling1=tf.keras.layers.MaxPooling2D()

    conv2=tf.keras.layers.Conv2D(32,3, padding="same", activation="relu")
    pooling2=tf.keras.layers.MaxPooling2D()

    conv3=tf.keras.layers.Conv2D(64,3, padding="same", activation="relu")
    pooling3=tf.keras.layers.MaxPooling2D()

    flatten=tf.keras.layers.Flatten()

    dense1=tf.keras.layers.Dense(512,activation="relu")
    dense2=tf.keras.layers.Dense(2, activation="softmax")

    out=conv1(inputs)
    out=pooling1(out)

    out=conv2(out)
    out=pooling2(out)

    out=conv3(out)
    out=pooling3(out)

    out=flatten(out)

    out=dense1(out)

    out=dense2(out)

    return tf.keras.Model(inputs=inputs, outputs=out)

def ResidualBlock(input_shape):
    inputs=tf.keras.layers.Input(shape=input_shape) #[32,32,128]

    # 在没有跳连接的层里，步长一半         
    conv1=tf.keras.layers.Conv2D(input_shape[2], 3, padding="same", activation="relu") #[16,16,128]
    conv2=tf.keras.layers.Conv2D(input_shape[2], 3, padding="same") #[16,16,128]

    relu=tf.keras.layers.ReLU()

    y=inputs

    #主通道
    y=conv1(y)
    y=conv2(y)
    y=y+inputs
    #激活
    y=relu(y)

    return tf.keras.Model(inputs=inputs, outputs=y)

def make_ResNet18_model(input_shape):

    inputs=tf.keras.layers.Input(input_shape) #[128,128,3]

    conv1=tf.keras.layers.Conv2D(64,7,strides=2,padding="same", activation="relu") #[64,64,64]

    pooling1=tf.keras.layers.MaxPooling2D() #[32,32,64]

    res64_1=ResidualBlock([input_shape[0]/4,input_shape[1]/4, 64]) #[32,32,64]
    res64_2=ResidualBlock([input_shape[0]/4,input_shape[1]/4, 64]) #[32,32,64]

    res128_1_1=tf.keras.layers.Conv2D(128,3,strides=2, padding="same", activation="relu");
    res128_1_2=tf.keras.layers.Conv2D(128,3, padding="same", activation="relu")

    res128_2=ResidualBlock([input_shape[0]/4,input_shape[1]/4, 128])

    res256_1_1=tf.keras.layers.Conv2D(256,3,strides=2, padding="same", activation="relu")
    res256_1_2=tf.keras.layers.Conv2D(256,3, padding="same", activation="relu")
    res256_2=ResidualBlock([input_shape[0]/8,input_shape[1]/8, 256])

    res512_1_1=tf.keras.layers.Conv2D(512,3,strides=2, padding="same", activation="relu")
    res512_1_2=tf.keras.layers.Conv2D(512,3, padding="same", activation="relu")
    res512_2=ResidualBlock([input_shape[0]/16,input_shape[1]/4, 512])

    pooling2=tf.keras.layers.MaxPooling2D()

    dense1=tf.keras.layers.Dense(512, activation="relu")
    dense2=tf.keras.layers.Dense(2, activation="softmax")

    y=inputs

    y=conv1(y)
    y=pooling1(y)
    y=res64_1(y)
    y=res64_2(y)
    y=res128_1_1(y)
    y=res128_1_2(y)
    y=res128_2(y)
    y=res256_1_1(y)
    y=res256_1_2(y)
    y=res256_2(y)
    y=res512_1_1(y)
    y=res512_1_2(y)
    y=res512_2(y)
    y=pooling2(y)
    y=tf.keras.layers.Flatten()(y)
    y=dense1(y)
    y=dense2(y)

    return tf.keras.Model(inputs=inputs, outputs=y)

r=make_ResNet18_model([128,128,3])

r.summary()





