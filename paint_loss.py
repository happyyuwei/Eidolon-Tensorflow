"""
鉴于本Linux服务器没有图形界面，无法直接显示损失函数，
目前就用最轻便的办法，保存成图像远程查看。
"""

from eidolon import train_tool

import platform

#获取当前系统
system = platform.system()

#只有在windows平台才默认显示，否则不会显示界面，而是保存成图片。
if system=="Windows":
    train_tool.paint_loss("log/train_log.txt")
else:
    train_tool.paint_loss("log/train_log.txt", save=True)