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
from tensorflow.keras.applications.vgg16 import VGG16

model = VGG16(weights='imagenet', include_top=False)

img=plt.imread("C:\\Users\\happy\\Downloads\\1.png")[:,:,0:3]
img=tf.reshape(img,[1,256,256,3])
# img=tf.image.resize(img,[224,224])

# print(model.layers)
# model.summary()

layer_outputs=[layer.output for layer in model.layers[0:3]]
# print(layer_outputs)

m1=tf.keras.Model(inputs=model.input, outputs=model.layers[2].output)
m1.summary()
o=m1(img)
plt.imshow(o[0,:,:,61:64])
plt.show()
