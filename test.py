import logging
import sys
import numpy as np
import matplotlib.pyplot as plt

# logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(funcName)s - %(module)s - %(message)s")

# def hello():
#     logging.info("hello")
#     logging.error("error")

# from CriminalArt.load_celebA import create_label_images

# create_label_images("./data/celebA/train")

# x=plt.imread("./data/celebA/train/000001.png")
# print(x)

# from WMNetv2.model_use import encode_watermark_from_image,decode_watermark_from_tensor


# a=encode_watermark_from_image("./WMNetv2/watermark/wm_x128.png", 256, 256)

# print(a)
# plt.imshow(a[0])
# plt.show()
# b=decode_watermark_from_tensor(a, 128,128)
# print(b.shape)
# plt.imshow(b[0])
# plt.show()

# from eidolon import train_tool
# # print(plt.imread("./WMNetv2/watermark/wm_x128.png")[:,:,0:3])

# a=train_tool.read_image("./WMNetv2/watermark/wm_binary_x128.png", 128,128, binary=True, threshold=0.5)
# plt.imshow(a[0])
# plt.show()