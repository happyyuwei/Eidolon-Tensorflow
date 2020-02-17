import numpy as np
import matplotlib.pyplot as plt

# im=plt.imread("./WMNetv2/watermark/wm_binary_x64.png")[:,:,0:3]
# im[im<=0.5]=0
# im[im>0.5]=1
# temp=np.zeros(np.shape(im))
# temp[:,:,0]=im[:,:,0]
# temp[:,:,1]=im[:,:,0]
# temp[:,:,2]=im[:,:,0]
# plt.imshow(temp)
# plt.show()
# plt.imsave("y.png",temp)

import tensorflow as tf
# from tensorflow.keras.applications.vgg16 import VGG16

# model = VGG16(weights='imagenet', include_top=False)

# img=plt.imread("C:\\Users\\happy\\Downloads\\1.png")[:,:,0:3]
# img=tf.reshape(img,[1,256,256,3])
# # img=tf.image.resize(img,[224,224])

# # print(model.layers)
# # model.summary()

# # layer_outputs=[layer.output for layer in model.layers[0:3]]
# # # print(layer_outputs)

# m1=tf.keras.Model(inputs=model.input, outputs=model.layers[18].output)
# m1.summary()
# o=m1(img)
# plt.imshow(o[0,:,:,61:64])
# plt.show()
# model=tf.keras.models.load_model("../face.h5")

# a=plt.imread("../1.jpg")
# a=tf.image.resize(a, [256,256])

# a=a/127.5-1
# a=tf.reshape(a, [1,256,256,3])

# b=model(a)

# b=b[0]*0.5+0.5

# plt.imshow(b)
# plt.show()


# from CriminalArt.load_celebA import create_labels

# # print(load_labels("./data/celebA/list_attr_celeba.txt")["000001.jpg"])
# create_labels("./data/celebA/test","./data/celebA/list_attr_celeba.txt")
# a=tf.io.read_file("./data/celebA/train/000001.txt")
# # print(a)
# a=tf.strings.split(a, sep=" ")
# # print(a)
# record_defaults = list([0.0] for i in range(1))
# print(record_defaults)
# a=tf.io.decode_csv(a, record_defaults=record_defaults)
# print(a)

# from CriminalArt.load_celebA import load_dataset
# from eidolon import config
# c=config.ConfigLoader()
# c.data_dir="./data/celebA"
# c.batch_size=1
# d=load_dataset(c, is_training=True)
# for a, b in d:
#     # print(a)
#     print(b)


# from CriminalArt import evaluate

# f=[float(each) for each in "0.0 1.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 1.0 0.0 1.0 0.0 0.0 1.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 1.0 0.0 1.0 0.0 1.0 0.0 0.0 1.0".split(" ")]
# f=np.array(f)

# # image=plt.imread("./data/celebA/train/000001.jpg")

# img=evaluate.create_visual(f)
# # print(img)

# plt.imshow(img)
# plt.show()
from eidolon import loader
from eidolon import config
from hide import secret_load
import os
config_loader = config.ConfigLoader()
config_loader.data_dir="./data/animate-face"
config_loader.image_height = 256
config_loader.image_width = 256

train_loader = loader.ImageLoader(os.path.join(
    config_loader.data_dir, "train"), is_training=True)

train_dataset = train_loader.load(config_loader, load_function=secret_load.load_image)

for a,b in train_dataset:
    print(a)
    print(b)

