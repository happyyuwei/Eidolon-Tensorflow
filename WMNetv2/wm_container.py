"""
该脚本定义水印训练，继承至pixel-to-pixel
"""

import pixel_container

class WMContainer(pixel_container.PixelContainer):

    def __init__(self, config_loader):
        # 调用父类
        super(WMContainer, self).__init__(config_loader)
    
    def on_prepare(self):
        """
        准备阶段，完成以下事宜：
        1. 调用父类的方法，该方法完成以下事宜：
            1. 加载数据集
            2. 创建网络与优化器
            3. 将网络与优化器注册到父类中，以便自动保存
            4. 调用父类on_prepare
        2. 创建水印提取网络
        """
        # 调用父类
        super(WMContainer, self).on_prepare()
