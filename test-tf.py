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

# import matplotlib.pyplot as plt

# a=plt.imread("198_GT.png")
# b=plt.imread("198_PR.png")

# print(np.average(np.abs(a-b)))


from eidolon import loader
from eidolon import config
from hide import secret_load
import os
config_loader = config.ConfigLoader()
config_loader.data_dir="./data/animate-face"
config_loader.image_height = 256
config_loader.image_width = 256

# train_loader = loader.ImageLoader(os.path.join(
#     config_loader.data_dir, "train"), is_training=True)

# train_dataset = train_loader.load(config_loader, load_function=secret_load.load_image)

# for a,b in train_dataset:
#     # plt.figure(1)
#     # plt.imshow(a[0,:,:,0:3]*0.5+0.5)
#     # plt.figure(2)
#     # plt.imshow(a[0,:,:,3:6]*0.5+0.5)
#     # plt.figure(3)
#     # plt.imshow(b[0]*0.5+0.5)
#     # plt.show()
#     pass

config_loader.data_dir="./hide/img/"
config_loader.image_type="png"
secret_loader = loader.ImageLoader(config_loader.data_dir, is_training=True)
secret_dataset=secret_loader.load(config_loader, load_function=secret_load.load_secret_image)
print(secret_dataset)
for x in secret_dataset.take(1):
    print(x)
