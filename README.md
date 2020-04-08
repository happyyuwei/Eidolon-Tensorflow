# Eidolon-Tensorflow（New version coming Soon）
 A framework to manage the training lifecycle of tensorflow apps.

# What's Eidolon?

![avatar](./instructions/top.png)






# APIs(使用说明尚未整理)
## `loader.read_image`
### 签名
    read_image(
        image_file, 
        image_type
    )

### 例子
    # 该例子用于读取一张名为color.jpg的图片。
    # 图片类型是jpg。
    read_image("color.jpg","jpg")

### 参数
* `image_file`: 字符串，待读取的图片名。
* `image_type`: 字符串，图片类型。目前支持：png, jpg, bmp.

### 返回值
* 图像3-D张量 `[img_height, img_width, channel]`               


# Inner Function
## lifecycle
整个训练的生命周期分为三部分组成：准备阶段，训练阶段，结束阶段，分别对应以下三个函数：
* on_prepare( ) 
    > 准备阶段，该方法在最初调用。用于完成所有准备工作，例如：载入数据集，创建模型，指定优化器等。
* on_train( )
    > 正式训练阶段，在完成on_prepare( )后调用该方法。持续对模型进行迭代。 
* on_finish( )
    > 训练结束后调用，在结束on_train( )方法后调用。 用于做一些最后的整理工作，例如测试最终模型效果，保存最终模型等。







