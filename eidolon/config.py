"""
@update 2019.12.20
在新版本Eidolon中移除txt格式的配置文件

随着配置参数越来越多，本人计划重新设计配置文件，先计划如下：
1. 使用json更好管理配置逻辑，现在的键值对key=value方式有些力不从心。
2. 重新设计配置类，使得添加额外配置更方便
2. 增加一个轻量级配置界面，使得每次修改时更加清晰不会出错。
3. 配置接口依旧适用LoadConfig类， 完全兼容旧版本配置逻辑。
4. 在过渡期间将把所有旧应用配置文件转成json,在此期间旧版本配置依旧兼容，但绝对不推荐在新应用中使用旧版本配置，在过渡期后，旧版本配置代码将被移除。
5. 访问设置推荐使用config_loader.config, 该对象为字典，可以直接使用字典方式访问， 如：config_loader.config["dataset"]["buffer_size"]["value"]。
不排除在今后阶段移除写法 config_loader.buffer_size的可能。
@since2019.11.30
@author yuwei
"""

import json

default_config = {

    "keys": ["dataset", "augmentation", "training", "pixel2pixel"],

    "dataset": {
        "keys": ["buffer_size", "batch_size", "image_size", "image_type", "input_number"],

        "buffer_size": {
            "value": 0,
            "desc": "载入训练图像时设置的缓存区大小，若设置为0，则一次性载入所有图像。（在训练数据量不大的时候推荐设置为0）"
        },
        "batch_size": {
            "value": 1,
            "desc": "训练时一次输入的图片张数，在生成网络训练中，推荐每次输入一张图像。"
        },
        "image_size": {
            "value": [256, 256, 3],
            "desc": "输入图像的尺寸，依次为：宽、高、通道。数据集中所有图像会裁剪成该尺寸。（注意：多余通道会被直接忽略）。"
        },
        "image_type": {
            "value": "jpg",
            "desc": "图像格式，可选：jpg, png, bmp。"
        },
        "input_number": {
            "value": 1,
            "desc": "单次输入图片数量，部分应用需要输入多张图片才能输出一张图片。默认输入一张。"
        }
    },
    "augmentation": {
        "keys": ["data_flip", "crop_size"],

        "data_flip": {
            "value": 0,
            "desc": "图像反转概率，在输入图像的时候存在一定的概率翻转图像已达到数据增强的效果。可输入（0-1）的小数。默认为0（不翻转）。"
        },
        "crop_size": {
            "value": [0, 0],
            "desc": "图像裁剪，在输入图像的时候随机裁剪一定的尺寸已达到数据增强。默认情况下关闭。"
        }
    },
    "training": {
        "keys": ["epoch", "save_period", "data_dir", "checkpoints_dir", "log_dir", "tensorboard_enable", "training_device", "remove_history_checkpoints", "load_latest_checkpoints"],
        "container": {
            "value": "eidolon.pixel_container.PixelContainer",
            "desc": "定义训练使用的容器。容器用于管理整个训练的生命周期。"
        },
        "epoch": {
            "value": 2000,
            "desc": "训练轮数。默认为2000轮。"
        },
        "save_period": {
            "value": 1,
            "desc": "保存周期。每过一个保存周期，将会保存训练的检查点以及记录训练日志。"
        },
        "data_dir": {
            "value": "../../data/",
            "desc": "训练数据集所在路径，相对位置为当前应用app所在位置。"
        },
        "checkpoints_dir": {
            "value": "./training_checkpoints/",
            "desc": "保存检查点路径，相对位置为当前应用app所在位置。"
        },
        "log_dir": {
            "value": "./log/",
            "desc": "保存训练日志路径，相对位置为当前应用app所在位置。"
        },
        "tensorboard_enable": {
            "value": False,
            "desc": "启用tensorboard记录日志。可以使用tensorboard在页面查看训练情况。"
        },
        "training_device": {
            "value": "default",
            "desc": "训练使用设备，若需要指定设备，请完整指定设备编号，如：/CPU:0，/GPU:0。若使用默认设备，输入default"
        },
        "remove_history_checkpoints": {
            "value": True,
            "desc": "则只保留最近的训练检查点。若启用该设置，早期检查点均会被移除且无法复原。（注意：若禁用该设置，则所有保存点会被保留，会占用大量磁盘空间。）"
        },
        "load_latest_checkpoints": {
            "value": True,
            "desc": "载入最近的检查点。若关闭该设置，将从头开始训练。"
        }
    },
    "pixel2pixel": {
        "keys": ["generator", "discriminator", "high_performance", "training_callback", "callback_args", "lambda"],
        "generator": {
            "value": "unet",
            "desc": "训练使用的生成器结构，默认使用Unet。包含选择：unet, resnet16。"
        },
        "discriminator": {
            "value": "no",
            "desc": "训练使用的生成器结构，默认不使用判决器。包含选择：no, gan, cgan。"
        },
        "high_performance": {
            "value": False,
            "desc": "如果在低配GPU中(<4G)，可能发生网络结构过于复杂而显存不足的情况。禁用该选项时，会把UNet中的编码器最后一层与解码器第一层去除。"
        },
        "callback_args": {
            "value": ["--decoder=../../trained_models/models/auto_mnist_x64", "--wm_path=../../WMNetv2/watermark/wm_binary_feature_x64.png"],
            "desc": "设置回调函数输入参数，用于初始化自定义的回调参数。目前支持--key=value格式。输入内容为数组，每个元素请用逗号隔开。目前有效参数：--decoder, --attack, --wm_path。"
        },
        "lambda": {
            "value": [1, 1],
            "desc": "设置损失函数超参数。输入内容为数组，每个元素请用空格隔开。"
        }
    }
}


def create_config_JSON_temple(config_dir):
    """
    创建JSON配置文件模板。
    """
    # 生成JSON字符串
    json_str = json.dumps(default_config, ensure_ascii=False, indent=2)
    # 写入文件
    with open(config_dir, "w", encoding="utf-8") as f:
        f.write(json_str)


class ConfigLoader:

    def __init__(self, config_dir=None):
        """
        @update 2019.12.30
        移除对旧版本的兼容
        @update 2019.11.30
        配置文件2.0， 兼容旧版本。
        @update 2019.11.27
        允许创建空的Loader
        @author yuwei
        @since 2018.12
        :param config_dir:
        """
        # 目前均会启用旧属性兼容
        old_attribate_enable = True

        if config_dir == None:
            # set default params
            self.config = default_config
        else:
            # load from file
            """
            如果配置文件是JSON,则调用json解析，否则为旧版本配置文件
            以上做法是为了兼容旧版本，已经不推荐使用旧版本设置配置文件。
            旧版本的配置文件将在今后移除，过度阶段后将不在支持。
            @since 2019.11.30
            @author yuwei
            """
            with open(config_dir, "r", encoding="utf-8") as f:
                self.config = json.load(f)

        if old_attribate_enable == True:
            # 兼容旧版本访问方式，旧属性目前依旧可以访问
            # buffer size
            self.buffer_size = self.config["dataset"]["buffer_size"]["value"]
            # batch size
            self.batch_size = self.config["dataset"]["batch_size"]["value"]
            # image width
            self.image_width = self.config["dataset"]["image_size"]["value"][0]
            # image height
            self.image_height = self.config["dataset"]["image_size"]["value"][1]
            # channel
            self.image_channel = self.config["dataset"]["image_size"]["value"][2]
            # image type
            self.image_type = self.config["dataset"]["image_type"]["value"]
            # input number
            self.input_num = self.config["dataset"]["input_number"]["value"]
            # data flip
            self.data_flip = self.config["augmentation"]["data_flip"]["value"]
            # crop width
            self.crop_width = self.config["augmentation"]["crop_size"]["value"][0]
            # crop height
            self.crop_height = self.config["augmentation"]["crop_size"]["value"][1]

            # conatiner
            self.container = self.config["training"]["container"]["value"]
            # epoch
            self.epoch = self.config["training"]["epoch"]["value"]
            # save period
            self.save_period = self.config["training"]["save_period"]["value"]
            # data dir
            self.data_dir = self.config["training"]["data_dir"]["value"]
            # checkpoints
            self.checkpoints_dir = self.config["training"]["checkpoints_dir"]["value"]
            # log dir
            self.log_dir = self.config["training"]["log_dir"]["value"]
            #tensorboard enable
            self.tensorboard_enable=self.config["training"]["tensorboard_enable"]["value"]
            # training device
            self.training_device = self.config["training"]["training_device"]["value"]
            # remove history
            self.remove_history_checkpoints = self.config[
                "training"]["remove_history_checkpoints"]["value"]
            # load latest checkpoint
            self.load_latest_checkpoint = self.config["training"]["load_latest_checkpoints"]["value"]

            # pixel to pixel
            # generator
            self.generator = self.config["pixel2pixel"]["generator"]["value"]
            # discriminator
            self.discriminator = self.config["pixel2pixel"]["discriminator"]["value"]
            self.high_performance = self.config["pixel2pixel"]["high_performance"]["value"]
            # callback args
            self.callback_args = self.config["pixel2pixel"]["callback_args"]["value"]
            # parse lambda array
            self.lambda_array = [
                float(x) for x in self.config["pixel2pixel"]["lambda"]["value"]]


class ArgsParser:
    """
    将输入参数转化成key-vale,F
    主要用于回调函数的参数中，支持--key=value格式
    @author yuwei
    @since 2019.12.6
    """

    def __init__(self, args):
        # 传入的args是字符串
        self.args_dict = {}

        if len(args) >= 1:
            for each in args:
                # 解析
                key, value = each.split("=")
                # 存储
                self.args_dict[key.replace("--", "")] = value

        print("callback args:{}".format(self.args_dict))

    def get(self, key):
        """
        获取参数，若不存在，返回None
        """
        try:
            return self.args_dict[key]
        except KeyError:
            return None
