from eidolon.pixel_container import PixelContainer
from eidolon.config import ArgsParser
from eidolon import train_tool

import tensorflow as tf


class StyleContainer(PixelContainer):
    """
    训练风格迁移的容器
    继承至父类pixe-to-pixel的容器
    callback_args: --style_image=... 风格图像的位置

    风格损失使用vgg16预训练网络，提取(第4,7,11,15,18下标为1)，(3,6,10,14,18下标为0)
    若输入尺寸为256，则每一个激活层输出为128，64，32，16，8
    """

    def on_prepare(self):
        """
        准备阶段，完成如下事件：
        1. 加载vgg16风格损失网络
        2. 加载风格损失图片
        """

        # 获取输入参数
        args_parser = ArgsParser(self.config_loader.callback_args)
        # 获取风格图像位置
        style_path = args_parser.get("style_image")

        # 加载风格图像
        self.style_image = train_tool.read_image(
            style_path, self.config_loader.image_width, self.config_loader.image_height, change_scale=True)

        # 加载vgg16网络，若网络不存在，会直接经行下载
        # 该网络使用image-net训练，不加载全连接层
        vgg_16 = tf.keras.applications.vgg16.VGG16(
            weights='imagenet', include_top=False)

        # 激活层id
        activate_id = [3, 6, 10, 14, 18]
        self.activate_list = []
        # 生成各个激活层
        for id in activate_id:
            self.activate_list.append(tf.keras.Model(
                inputs=vgg_16.input, outputs=vgg_16.layers[id].output))

        #由于风格图像给定，每一层特征就给定，可以提前计算
        #直接计算gram举证
        self.style_feature_list=[]
        for activate in self.activate_list:
            self.style_feature_list.append(self.gram(activate(self.style_image)))

        # 执行父类方法
        super(StyleContainer, self).on_prepare()

    def gram(self, mat):
        """
        格拉姆举证
        """
        return tf.matmul(mat,tf.transpose(mat))

    def style_loss(self, image):
        """
        风格损失，计算图像每一个激活层的Grim矩阵
        """
        loss=0
        #多层的损失
        for i in range(len(self.activate_list)):
            #提取特征
            image_style=self.activate_list[i](image)
            #计算风格损失
            loss=loss+tf.reduce_mean(tf.square(self.gram(image_style)-self.style_feature_list[i]))
        
        return loss

    
    def content_loss(self, image, target):
        """
        内容损失,只计算第7层（下标6）的特征,存在列表第二个位置
        """
        image_feature=self.activate_list[2](image)
        target_feature=self.activate_list[2](target)

        return tf.reduce_mean(tf.square(image_feature-target_feature))


    def compute_loss(self, input_image, target):
        """
        重写损失函数
        """
        # 计算生成网络输出图像
        gen_output = self.generator(input_image, training=True)

        #计算风格损失
        style_loss=self.style_loss(gen_output)

        #计算内容损失
        content_loss=self.content_loss(gen_output, target)

        #总损失
        loss= style_loss+0.8*content_loss

        #损失集合
        loss_set={
            "total_gen_loss":loss,
            "style_loss":style_loss,
            "content_loss":content_loss
        }

        return loss_set
