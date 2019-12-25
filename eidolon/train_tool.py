#third part lib
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#system lib
import os
import time
import math

"""
image type, this is used to define whether the show image is binary or color
@author yuwei
@since 2019.11.25
"""
# 彩色图像
IMAGE_RGB = "rgb"
# 二值图像
IMAGE_BINARY = "binary"


def save_image(image_tensor, file, min=0, max=1, image_type=IMAGE_RGB):
    """
    a general function to save image tensor, the tensor dim is [1,height,width,3] with pixel [0,1]

    @update 2019.12.25
    该函数底层调用plt.imsave()函数保存图片，在linux上大量保存存在不稳定的情况，Runtime Error, libpng signaled error.
    尚未找出原因，目前尝试换成PIL库保存。

    @since 2019.9.12
    @author yuwei
    :param image_tensor:
    :param file
    :param min
    :param max
    :param image_type 即将移除（？） @yuwei 2019.11.29
    :return:
    """
    # the input is 4-dim tensor
    image_tensor = image_tensor[0]
    # if the scale is already [0,1], do not need to change scale
    if min != 0 or max != 1:
        # change scale to [0,1]
        # e.g.1 [-1,1]-(-1)=[-1,1]+1=[0,2] =>[0,2]/2=[0,1]
        # e.g.2  [1,2]-1=[0,1] =>[0,1]/(-1+2)=[0,1]
        image_tensor = (image_tensor - min) / (-min + max)

    # to numpy array
    drawable = np.array(image_tensor)
    drawable[drawable < 0] = 0
    drawable[drawable > 1] = 1

    [height, width, channel] = np.shape(drawable)
    # change binary image
    if image_type == IMAGE_BINARY and channel >= 3:
        drawable = drawable[:, :, 0]+drawable[:, :, 1]+drawable[:, :, 2]
        drawable = drawable/3
        level = (min+max)/2
        # filter
        drawable[drawable >= level] = 1
        drawable[drawable < level] = 0
        temp = np.zeros([height, width, 3])
        temp[:, :, 0] = drawable
        temp[:, :, 1] = drawable
        temp[:, :, 2] = drawable
        drawable = temp

    # if only one channel, then make it to rgb change with the same
    # @author yuwei
    # @since 2019.9.17
    if channel == 1:
        # to three chanel
        temp = np.zeros([height, width, 3])
        temp[:, :, 0] = drawable[:, :, 0]
        temp[:, :, 1] = drawable[:, :, 0]
        temp[:, :, 2] = drawable[:, :, 0]
        drawable = temp

    # save
    # plt.imsave(file, drawable)
    #跟换PIL库保存图片
    drawable=drawable*255
    img = Image.fromarray(drawable.astype('uint8'))
    img.save(file,"png")


def save_images(image_list, title_list, image_dir, seq=""):
    """
    this function is to save image list, each image in he image list should be tensor with [0, height,width, 3]
    @since 2019.9.14
    @author yuwei
    :param image_list:
    :param title_list:
    :param image_dir
    :param seq
    :return:
    """
    if seq != "":
        seq = str(seq) + "_"

    for i in range(len(title_list)):
        dir = os.path.join(image_dir, seq + title_list[i] + ".png")
        save_image(image_list[i], dir, min=-1, max=1)


def read_image(path, width, height, change_scale=False, binary=False):
    """
    按指定大小读取图片，注意：如果尺寸和实际尺寸不符合，会进行裁剪或者填充平铺以满足指定尺寸。
    @update 2019.11.27
    从train_watermark中将该函数移至此处，作为基础设施

    this function will return the image
    @update 2019.11.26
    修复bug:在输入tensorflow时，没有对其进行尺度变换至[-1,1]

    @since 2019.9.12
    @author yuwei
    :param path:
    :param width:
    :param height:
    :param_scale: false:不进行任何变化,[0,1]，True:变换到[-1,1]
    :param binary: 将图像变为二值图
    :return:
    """
    image = plt.imread(path)
    image = image[:, :, 0:3]
    image = np.resize(image, [1, height, width, 3])

    if binary == True:
        # filter
        image[image < 0.5] = 0
        image[image >= 0.5] = 1
        # same in each channel
        image[0, :, :, 1] = image[0, :, :, 0]
        image[0, :, :, 2] = image[0, :, :, 0]

    # scale to [-1,1]
    if change_scale == True:
        image = image*2-1

    # conver to tensor
    image = tf.convert_to_tensor(image)
    return image


def parse_last_epoch(file):
    """
    parse last epoch, invoked by init function
    :param file:
    :return:
    """
    if os.path.exists(file) is False:
        return 0
    else:
        with open(file, "r") as f:
            lines = f.readlines()
            # 修复读取空文件报错的bug
            # @since 201.912.21
            if len(lines) == 0:
                return 0
            else:
                last_line = lines[-1]
                return int(last_line.split(",")[0].split("=")[1])


"""
actually, log epoch must be vary instead of 1
@fixed at version 0.93
@author yuwei
@since 2019.1.17

"""


class LogTool:

    def __init__(self, log_dir, save_period, tensorboard_enable=False):
        """
        initial directionary
        负责管理整个训练系统的记录
        目前支持默认文本方式记录，与tensorboard方式记录。
        用户可以选择启用或者禁用tensorboard方式记录，而文本方式记录无法禁用。
        目前默认将tensorboard日志记录在日志目录的tensorboard文件夹中
        @update 2019.12.24
        @author yuwei
        支持以tensorboard的规定格式输出

        :param log_dir:
        :param save_period
        :param tensorboard_enable
        """
        self.log_dir = log_dir
        self.save_period = save_period
        self.image_dir = os.path.join(log_dir, "result_image")
        self.train_log_dir = os.path.join(log_dir, "train_log.txt")
        # create dir if not exist
        if os.path.exists(self.log_dir) is False:
            os.mkdir(log_dir)
        if os.path.exists(self.image_dir) is False:
            os.mkdir(self.image_dir)
        # upate current epoch
        # fix bug
        # @since 2019.11.23
        # @author yuwei
        self.current_epoch = parse_last_epoch(self.train_log_dir) + 1

        # beta @since 2019.12.24 支持tensorboard
        # 目前默认将tensorboard日志记录在日志目录的tensorboard文件夹中
        self.tensorboard_enable=tensorboard_enable
        print("Tensorboard enable: {}".format(tensorboard_enable))
        if tensorboard_enable == True:
            self.tensorboard = tf.summary.create_file_writer(
                os.path.join(self.log_dir, "tensorboard"))

    def save_image_list_tensorboard(self, image_list, title_list):
        """
        调用 tensorflow 的tf.summary.image保存测试图片
        需要使用tensorboard在网页端查看输出内容。
        @since 2019.12.24
        @ author yuwei
        """
        # 测试支持tensorflow
        with self.tensorboard.as_default():
            for i in range(len(title_list)):
                # The input image of tensorboard is 4 dim, the first dim is batch.
                # by default,at most 3 images in a batch will be shown
                tf.summary.image(
                    title_list[i], image_list[i]*0.5+0.5, step=self.current_epoch)

    def save_loss_tensorboard(self, loss_set):
        """
        使用tensorboard保存损失函数，调用tf.summary.scalar()
        可以使用tensorboard在网页端查看变化曲线
        """
        # 测试支持tensorboard
        with self.tensorboard.as_default():
            for key in loss_set:
                tf.summary.scalar(
                    key, data=loss_set[key], step=self.current_epoch)

    def save_image_list(self, image_list, title_list):
        """
        @update 2019.11.27
        the basic logis removes to extern file
        @since 2019.9.14
        @author yuwei
        :param image_list:
        :param title_list:
        :return:
        """
        if self.tensorboard_enable==True:
            save_images(image_list, title_list, self.image_dir, self.current_epoch)
        

    def save_loss(self, loss_set):
        """
        update the save keys with all kinds.
        @author yuwei
        @update 2019.9.13
        save loss
        :param gen_loss:
        :param disc_loss:
        :return:
        """
        with open(self.train_log_dir, "a") as f:
            # combine conetent
            content = ""
            for key in loss_set:
                content = content + ",{}={}".format(key, loss_set[key])

            # create line
            line = "epoch={},timestamp={}{}\n".format(self.current_epoch,
                                                      time.strftime("%b-%d-%Y-%H:%M:%S", time.localtime()), content)
            # change line
            f.writelines(line)
        
        if self.tensorboard_enable==True:
            self.save_loss_tensorboard(loss_set)


    def plot_model(self, model, model_name):
        """
        绘画模型结构,需要安装pydot.
        模型会保存在日志文件的更目录下。
        """
        """
        需要注意参数rankdir，官方解释如下：
        rankdir argument passed to PyDot, 
        a string specifying the format of the plot: 'TB' creates a vertical plot; 
        'LR' creates a horizontal plot.
        """
        file = os.path.join(self.log_dir, "{}.png".format(model_name))
        tf.keras.utils.plot_model(
            model, to_file=file, show_shapes=True, rankdir="TB", dpi=64)

    def update_epoch(self):
        """
        update epoch
        :return:
        """
        # next epcoh
        self.current_epoch = self.current_epoch + 1


def parse_log(log_dir):
    """
    parse log, current includes generator loss and disc loss
    @since 2019.2

    at present, different kinds of loss can all be included
    the format is still: epoch=1,timestamp=...,loss1=...,loss2=...
    @update 2019.9.12
    @author yuwei
    :param log_dir:
    :return:
    """
    with open(log_dir, "r") as f:
        lines = f.readlines()

    # parse all the keys in the log
    loss_map = dict()
    for line in lines:
        line = line.strip().split(",")
        for i in range(2, len(line)):
            key, value = line[i].split("=")
            array = loss_map.get(key)
            if array is None:
                array = []
                loss_map[key] = array
            array.append(float(value))

    return loss_map


def paint_loss(log_dir, update_time=30, save=False):
    """
    @update 2019.12.24
    临时功能，若save=True，则不会显示而是直接保存。
    此功能用于未安装图形界面的linux设备。
    再一次吐槽plt的实时显示功能鸡肋，完全不符合前端展示的理念。

    @update 2019.11.25
    Adding dymanic show panel. (under testing)
    使用plt.interactive交互技术，可以动态显示损失曲线，无需每次查看前手动重启程序
    matplotlib这一点设计的还是不够好，需要入侵非常多的代码才能做到动态显示。(╯‵□′)╯︵┻━┻
    print loss
    change paint loss
    @update 2019.9.13
    @author yuwei
    :param log_dir:
    :param update_time: default is 30s
    :return:
    """
    # parse logs
    loss_map = parse_log(log_dir)
    # calcualte how to show all the figures
    size = len(loss_map)
    # each row has at most three figures
    col = 3
    row = math.ceil(size / col)
    # if only one row, the column number is size number instead of 3
    if row == 1:
        col = size

    # paint
    plt.figure("Loss")
    # 如设置为不保存，则直接实时动态显示损失曲线
    if save==False:
        plt.ion()
        while True:
            num = 1
            for key in loss_map:
                # locate
                ax = plt.subplot(row, col, num)
                plt.title(key)
                # paint
                ax.plot(loss_map[key], "b")
                num = num + 1
                # update map every 30s
                loss_map = parse_log(log_dir)
            plt.pause(update_time)
        # display
        plt.ioff()
        plt.show()
    else:
        #若设置保存，则不会显示，而是直接保存成图片。
        while True:
            num = 1
            for key in loss_map:
                # locate
                ax = plt.subplot(row, col, num)
                plt.title(key)
                # paint
                ax.plot(loss_map[key], "b")
                num = num + 1
                # update map every 30s
                loss_map = parse_log(log_dir)
            #默认保存在应用目录的loss.png下，会自动覆盖。
            plt.savefig("loss.png")
            print("update...")
            time.sleep(update_time)


def remove_history_checkpoints(dir):
    """
    remove history checkpoints and remain nothing
    :param dir:
    :return:
    """
    if os.path.exists(dir) is False:
        return
    # get all checkpoints
    file_list = []
    for file in os.listdir(dir):
        file_list.append(file)

    # @since 2018.12.29 this method will remove all the checkpoints
    # sort
    # file_list.sort(reverse=True)

    # if only a check point is in the directionary, do not delete it
    # if len(file_list)<=3:
    #   return

    for each in file_list:
        # do not remove checkpoint file
        # if file_list[i] == "checkpoint":
        #    continue

        # remove history
        remove_file = os.path.join(dir, each)
        os.remove(path=remove_file)
