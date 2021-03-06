# third part lib
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# system lib
import os
import time
import math
import json
import importlib

"""
image type, this is used to define whether the show image is binary or color
@author yuwei
@since 2019.11.25
"""
# 彩色图像
IMAGE_RGB = "rgb"
# 二值图像
IMAGE_BINARY = "binary"



def load_function(function_name):
    """载入函数实例,形如 tf.keras....

    @since 2020.5.16
    从不同地方抽离成公共函数

    Arguments:
        function_name {[type]} -- [description]
    """

    module_name = ".".join(function_name.split(".")[0:-1])
    function_name = function_name.split(".")[-1]

    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    return function


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
    shape = np.shape(drawable)

    # 只有长宽的单通道灰度图
    # @since 2020.4.10 支持单通道图像绘制
    # @author yuwei
    if len(shape) == 2:
        [height, width] = np.shape(drawable)
        drawable = np.reshape(drawable, [height, width, 1])

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
    # 跟换PIL库保存图片
    drawable = drawable*255
    img = Image.fromarray(drawable.astype('uint8'))
    img.save(file, "png")


def save_images(image_list, title_list, image_dir, seq="", max_save=5, description=None, desc_file=None):
    """
    @update 2020.5.16
    增加单次保存多批次图像的功能
    支持保存图片描述，在分类任务中，可以在描述中保存图片预测结果

    this function is to save image list, each image in he image list should be tensor with [0, height,width, 3]
    @since 2019.9.14
    @author yuwei
    :param image_list:
    :param title_list:
    :param image_dir
    :param seq
    :param max 每轮最大保存数量
    :description 描述，可以是任何结构，推荐字典结构
    :desc_file 描述文件
    :return:
    """
    if seq != "":
        seq = str(seq) + "_"
    # 按照文件名保存
    for i in range(len(title_list)):
        # 拼接地址
        save_dir_prefix = seq + title_list[i]
        # 获取图像数量
        shape = np.shape(image_list[i])
        num = shape[0]
        # 超出数量不予保存
        if num > max_save:
            num = max_save
        # 保存多批次
        for j in range(num):
            # if num > 1:
                # 文件格式  {epoch}_{title}_{num}.png
                # 只有一种格式，不允许使用其他格式。
                # 当只保存一张图，num=1，则格式为 {epoch}_{title}.png
            save_dir_file = save_dir_prefix+"_{}.png".format(j+1)
            # else:
            #     save_dir_file = save_dir_prefix+".png"
            save_dir = os.path.join(image_dir, save_dir_file)
            image_batch = image_list[i]
            # 保存图片
            save_image(image_batch[j:j+1, :, :, :], save_dir, min=-1, max=1)
            # 保存描述到文件中
            if description != None:
                temp={}
                #将字典中的numpy转成list
                for key in description:
                    array=np.array(description[key][j])
                    temp[key]=array.tolist()

                save_desc = {
                    # 保存格式 {epoch}_{num}
                    seq+str(j): temp
                }
                if os.path.exists(desc_file) == True:
                     # 读取原文件
                    with open(desc_file, "r", encoding="utf-8") as f:
                        total_desc = json.load(f)
                # 若文件不存在则创建
                else:
                    total_desc = {}
                # 追加
                total_desc.update(save_desc)
                # 生成json
                total_desc_json = json.dumps(
                    total_desc, ensure_ascii=False, indent=2)
                # 保存
                with open(desc_file, "w", encoding="utf-8") as f:
                    f.write(total_desc_json)


def read_image(path, width, height, change_scale=False, binary=False, threshold=0.5):
    """
    按指定大小读取图片，注意：如果尺寸和实际尺寸不符合，会进行裁剪或者填充平铺以满足指定尺寸。
    返回四维tensor
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
    :param threshold: 二值图像阈值，只有当binary设置成true，该参数才有意义
    :return:
    """
    image = plt.imread(path)
    image = image[:, :, 0:3]
    image = np.resize(image, [1, height, width, 3])

    if binary == True:
        # filter
        image[image < threshold] = 0
        image[image >= threshold] = 1
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

    def __init__(self, log_dir, save_period, max_save=5, tensorboard_enable=False):
        """
        @update 2020.5.16
        支持单次保存多批次图像

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
        :param max_save
        """
        # 日志目录
        self.log_dir = log_dir
        # 保存周期
        self.save_period = save_period
        # 图像结果目录
        self.image_dir = os.path.join(log_dir, "result_image")
        # 标签结果目录，在result_image中
        self.label_dir = os.path.join(self.image_dir, "label.txt")
        # 训练日志
        self.train_log_dir = os.path.join(log_dir, "train_log.txt")
        # 最大保存轮数
        self.max_save = max_save
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
        self.tensorboard_enable = tensorboard_enable
        print("Tensorboard enable: {}".format(tensorboard_enable))
        # 当第一次调用保存tensorboard的时候才创建tensorboard writer
        # 避免在创建运行环境（如调用GPU）前调用tensorflow的任何内容
        # 记录tensorboard是否加载
        self.tensorboard_init_state = False

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
        if self.tensorboard_init_state == False:
            # 创建tensorboard记录实例
            self.tensorboard = tf.summary.create_file_writer(
                os.path.join(self.log_dir, "tensorboard"))
            self.tensorboard_init_state = True

        # 测试支持tensorboard
        with self.tensorboard.as_default():
            for key in loss_set:
                tf.summary.scalar(
                    key, data=loss_set[key], step=self.current_epoch)

    def save_image_list(self, image_list, title_list, description=None):
        """
        @update 2019.11.27
        the basic logis removes to extern file
        @since 2019.9.14
        @author yuwei
        :param image_list:
        :param title_list:
        :param description: 推荐字典或者字符串
        :return:
        """

        save_images(image_list, title_list, self.image_dir,
                    self.current_epoch, self.max_save, description, self.label_dir)

        if self.tensorboard_enable == True:
            self.save_image_list_tensorboard(image_list, title_list)

    def save_label_list(self, predict_label, title_list, labels):
        """保存分类结果，用于分类任务测试可视化

        Arguments:
            predict_list {[预测结果]} -- [description]
            title_list {[标题，与分类图像标题对应]} -- [description]
            labels {[标签]}
        """
        # 解析

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

        if self.tensorboard_enable == True:
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
    if save == False:
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
        # 若设置保存，则不会显示，而是直接保存成图片。
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
            # 默认保存在应用目录的loss.png下，会自动覆盖。
            plt.savefig("loss.png")
            print("update...")
            time.sleep(update_time)


def visual_classify(img, label_list, prob_list):
    """
    可视化分类结果，把类别的概率展示出来。返回四维张量【-1,1】
    目前尚在设计之中
    :param: img 四维张量
    :param:label_list 标签集
    :param: prob_list 概率集
    """

    # 转成uint8类型
    img = np.uint8(img*127.5+127.5)
    img = Image.fromarray(img)
    # 获取尺寸
    w, h = img.size

    # 添加图片
    result = Image.new('RGB', (w, int(h*1.5)), (255, 255, 255))
    result.paste(img, box=(0, 0))

    draw = ImageDraw.Draw(result)

    # 生成标签
    word = ""
    for i in range(len(label_list)):
        # 概率保留两位
        word = word+label_list[i]+": "+"{:.2f}".format(prob_list[i])+"."

    # 添加标签
    draw.text((int(w*0.1), int(h*1.2)), word, fill="black")

    # 转 numpy
    result = np.array(result)
    [h, w, c] = np.shape(result)

    # 从【0,255】 转成 【-1,1】
    result = result/127.5-1

    # 转成四维张量
    return np.reshape(result, [1, h, w, c])



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
