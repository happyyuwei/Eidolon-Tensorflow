import logging
import sys
import numpy as np
import matplotlib.pyplot as plt

# a=np.zeros([4000,2]);
# a[0:2000,0]=1
# a[2000:,1]=1

# # np.savetxt("labels.txt", a, fmt="%.1f", delimiter=",")

# a="1"
# b,c=a
# print(b)
# print(c)
a = {}
# import os
# #
# print(os.path.isdir("2"))

# 1 0
# 0 1

# 1 1 0 0
# 1 1 0 0
# 0 0 1 1
# 0 0 1 1

# rand = np.random.randint(0,2,[2,2])
# rand=np.array([[1,0],[0,1]])
# print(rand)
# rand=np.repeat(rand,2, axis=1)
# rand=np.repeat(rand,2, axis=0)

# print(rand)

a = np.array([[1, 2], [3, 4]])
b = np.array([[1, 2], [3, 4]])

c = np.zeros([1, 2, 2, 2])
c[0, :, :, 0] = a
c[0, :, :, 1] = b

c = np.mean(c, axis=3)


print(128/8)
