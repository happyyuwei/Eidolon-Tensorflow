from eidolon.pixel_container import PixelContainer
from eidolon.config import ArgsParser

class StyleContainer(PixelContainer):
    """
    训练风格迁移的容器
    继承至父类pixe-to-pixel的容器
    callback_args: --style_image=... 风格图像的位置
    """

    def on_prepare(self):
        """
        准备阶段，完成如下事件：
        1. 加载vgg16风格损失网络
        2. 加载风格损失图片
        """

        #获取输入参数
        args_parser=ArgsParser(self.config_loader.callback_args)
        #获取风格图像
        # args

        #执行父类方法
        super(StyleContainer,self).on_prepare()
