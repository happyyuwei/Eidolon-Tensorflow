import numpy as np
import matplotlib.pyplot as plt

im=plt.imread("./WMNetv2/watermark/wm_binary_x64.png")[:,:,0:3]
im[im<=0.5]=0
im[im>0.5]=1
temp=np.zeros(np.shape(im))
temp[:,:,0]=im[:,:,0]
temp[:,:,1]=im[:,:,0]
temp[:,:,2]=im[:,:,0]
plt.imshow(temp)
plt.show()
plt.imsave("y.png",temp)

