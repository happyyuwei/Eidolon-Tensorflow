import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from CriminalArt import load_celebA

from eidolon import config


# model=tf.keras.models.load_model("./c.h5")
# # model.summary()
# img=plt.imread("test/000033.jpg")[:,:,0:3]

# # plt.imshow(img)
# # plt.show()

# img=img*2-1


# f=np.zeros([1,178,178,3])
# f[0]=img

# out=model(f, training=True)
# out=tf.sigmoid(out)
# out=np.array(out)
# # out[out<=0]=-1
# # out[out>0]=1

# print(out)

config_loader = config.ConfigLoader()
config_loader.image_width = 178
config_loader.image_height = 178
config_loader.data_dir = "./data/celebA"


test_dataset = load_celebA.load_dataset(
    config_loader, is_training=False)

model = tf.keras.models.load_model("./c.h5")

count = 0

error_total=np.zeros(40)

for input_image, target in test_dataset:
    label, _ = target
    out = model(input_image, training=True)
    out = tf.sigmoid(out)
    out=np.array(out)
    out[out>0]=1
    out[out<=0]=0

    error=tf.abs(out-label)

    error_total=error_total+error[0]

    count=count+1

    print("\r"+"i={}, err={}".format(count,1-np.array(error_total)/count), end="", flush=True)
