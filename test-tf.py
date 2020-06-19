import numpy as np
import matplotlib.pyplot as plt
import sys

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
import tensorflow_datasets as tfds
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


# from eidolon import loader
# from eidolon import config
# # from hide import secret_load
# import os
# config_loader = config.ConfigLoader()
# config_loader.data_dir="./data/animate-face"
# config_loader.image_height = 256
# config_loader.image_width = 256

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

# config_loader.data_dir="./hide/img/"
# config_loader.image_type="png"
# secret_loader = loader.ImageLoader(config_loader.data_dir, is_training=True)
# secret_dataset=secret_loader.load(config_loader, load_function=secret_load.load_secret_image)
# print(secret_dataset)
# for x in secret_dataset.take(1):
#     print(x)


# model=tf.keras.models.load_model("./app/car/model/generator.h5")

# list=[]
# for x in range(1,29):
#     list.append("{}.png".format(x))


# l=loader.ImageLoader("./data/car/test",is_training=False, file_list=list)
# d=l.load(image_type="png")


# i=1
# for x,_ in d:

#     y=model(x)

#     x=np.array(x[0])*0.5+0.5
#     x[x<0]=0
#     x[x>1]=1


#     y=np.array(y[0])*0.5+0.5
#     y[y<0]=0
#     y[y>1]=1
#     # print(y)
#     plt.imsave("./temp1/{}.png".format(i), x)
#     plt.imsave("./temp2/{}.png".format(i), y)
#     # plt.imshow(y)
#     # plt.show()
#     print(i)
#     i=i+1

# (train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
# train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
# train_images = (train_images - 127.5) / 127.5 # 将图片标准化到 [-1, 1] 区间内
# train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(10).batch(256)

# x=train_dataset.take(2)

# for a in x:
#     print(a.shape)

# a=tf.Variable([0.8,0.2])


# a[a>0.5]=1

# print(a)

# resize


# 压缩

# 偏置

# model=tf.keras.models.load_model("./app/test5/extractor.h5")
# a=plt.imread("./app/test5/test/147_PR.png")[:,:,0:3]

# # a=a/255.0

# plt.imshow(a)
# plt.show()

# img=tf.reshape(a,[1,128,128,3])

# img=img*2-1
# # plt.imshow(img[0])
# # plt.show()

# out=model(img, training=True)
# out=out[0]*0.5+0.5

# out=np.array(out)
# out[out>1]=1
# out[out<0]=0

# plt.imshow(out)
# plt.show()


# train_images, _, _, _ = tf.keras.datasets.mnist.load_data()

# a=tf.data.Dataset.list_files("C:\\Users\\happy\\Desktop\\x\\*.jpg",shuffle=False)
# for b in a:
#     print(b)

# from eidolon import loader, train_tool
# # a=loader.load_label_file("C:\\Users\\happy\\Desktop\\x\\labels.txt")
# # print(a)

# l=loader.ImageLoader("./data/cat-dog/train", is_training=True)
# d=l.load(load_function=loader.load_single_image,image_height=128,image_width=128, batch_size=2, load_label_function=loader.load_label_file)


# print(d.take(1))

# from eidolon.model import basic
# m=basic.make_Conv_models([128,128,3])

# for a,b in d.take(1):
#     pass


# print(a)
# print(tf.shape(a))
#     sys.exit()
# print(np.array(a).shape)

# from eidolon import train_tool

# a=train_tool.read_image("./app/5/log/result_image/105_PR.png",128,128)
# b=train_tool.read_image("./app/5/log/result_image/105_GT.png",128,128)
# p=tf.image.psnr(a,b, max_val=1)
# s=tf.image.ssim(a,b,max_val=1)
# print(p)
# print(s)

# (train_dataset, test_dataset), info=tfds.load(name="mnist", split=["train","test"], as_supervised=True,  with_info=True)


# def normalize_img(image, label):
#   """Normalizes images: `uint8` -> `float32`."""
#   return tf.cast(image, tf.float32) / 255., label

# train_dataset = train_dataset.map(
#     normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
# train_dataset = train_dataset.cache()
# train_dataset = train_dataset.shuffle(info.splits['train'].num_examples)
# train_dataset = train_dataset.batch(128)
# #预读取
# train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

# from eidolon import loader

# train_dataset, t=loader.load_dataset(preprocess_function=loader.process_image_classification, batch_size= 128,  data_dir="C:\\Users\\happy\\Desktop\\downloads\\",tfds_name="mnist", image_width=28, image_height=28,image_channel=1, crop_width=0, crop_height=0, flap_probability=0)
# train_dataset, t=loader.load_image_dataset(preprocess_function=loader.process_image_classification, batch_size= 128,  data_dir="C:\\Users\\happy\\Desktop\\downloads\\",tfds_name="mnist", image_width=28, image_height=28,image_channel=1, crop_width=0, crop_height=0, flap_probability=0)

# for a,b in train_dataset:
#     # print(a)
#     print(a)
#     sys.exit(0)

# (train_dataset, test_dataset), info = tfds.load(name="mnist",
#                                                         split=["train", "test"], as_supervised=True,  with_info=True)

# # print(info.splits['train'].num_examples)
# print(info)


# a=tf.cast(np.zeros([128,28,28,3]), tf.float32)
# s=tf.shape(a)
# print(np.array(s))


from eidolon import loader

# a=loader.load_custom_image_dataset(None, "C:\\Users\\happy\\Desktop\\Evil\\origin-data\\rio\\red",["rio_512_structure","rio_raw"], ["jpg","jpg"], 1, True, 128,128, 0, 0,0)

# print(a)

# for each in a:
#     print(each)
#     sys.exit(0)


def create_fuck_message(batch, message_len_sqrt, img_width):

    temp = np.zeros([batch, 128, 128, 3])
    repeat_len = img_width/message_len_sqrt

    for i in range(batch):
        rand = np.random.randint(0, 2, ([message_len_sqrt, message_len_sqrt]))
        rand = rand*2-1
        rand = np.repeat(rand, repeat_len, axis=1)
        rand = np.repeat(rand, repeat_len, axis=0)

        temp[i, :, :, 0] = rand
        temp[i, :, :, 1] = rand
        temp[i, :, :, 2] = rand

    temp = tf.cast(temp, tf.float32)

    return temp


# m = create_fuck_message(3, 16, 128)

# plt.imshow(m[0]*0.5+0.5)
# plt.show()
# plt.imshow(m[1]*0.5+0.5)
# plt.show()
# plt.imshow(m[2]*0.5+0.5)
# plt.show()
# plt.imshow(m[3]*0.5+0.5)
# plt.show()

from eidolon.tutorial.semantic.dataset import load_dataset
d=load_dataset("./data/city/train", 1)
print(d)
