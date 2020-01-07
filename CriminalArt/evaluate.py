import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont


import random

feature_labels = ["5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
                  "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair",
                  "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair",
                  "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes",
                  "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks",
                  "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat",
                  "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"]

font_theme = ["#0D3B66", "#F4D35E", "#EE964B", "#EA5A24"]


def create_visual(feature, width=1024, height=1024):
    """
    feature为[40]的0-1特征
    输出图片的大小默认为【1024,1024】
    """

    # 总共特征数
    num=0
    for each in feature:
        if each >0.5:
            num=num+1

    # 单个特征字符所占宽度
    each_width = int(height//num)

    # 创建图像模板
    # img=Image.new("RGB", (width, height), (250, 240, 202))
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    #字体相对路径是根据app下的具体应用为根目录的
    font = ImageFont.truetype(
        "../../CriminalArt/font/SF Arch Rival Extended.ttf", 40)

    current = 0
    for i in range(len(feature)):
        if feature[i] >0.5:
            rand = random.randint(0, len(font_theme)-1)
            draw.text((random.randint(0, int(width*2/3)), current), feature_labels[i].replace(
                "_", " "), font=font, fill=font_theme[rand])
            current = current+each_width

    print(num)
    return np.array(img)

def create_visual_tensor(feature, width=1024, height=1024):
    """
    创建视觉效果，输入tensor, 维度：【1，40】
    返回tensor，维度【1,1024,1024,3】
    """
    img=create_visual(feature[0], width, height)

    img=tf.cast(img, tf.float32)
    #变成[-1,1]
    img=img/127.5-1
    #转成四维tensor
    img=tf.reshape(img, [1, height, width, 3])

    return img





