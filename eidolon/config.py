"""
@update 2020.5.14
移除对txt格式配置的支持。
移除旧版json格式支持。
新版json配置字段，支持更全面配置方案。
增加字段访问接口，不在推荐旧版访问方式，过于冗长，如：config_loader.config["dataset"]["buffer_size"]["value"]。

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
import getopt
import sys
import os

# 默认配置文件
default_config = {

    "keys": ["runtime", "dataset", "augmentation", "training", "pixel"],

    "runtime": {
        "keys": ["path"],

        "path": {
            "value": [],
            "desc": "运行环境路径，若代码存放在非当前路径或者根路径，则需要将该路径添加，否则无法运行代码。"
        }
    },

    "dataset": {
        "keys": ["tensorflow_dataset", "data_dir", "buffer_size", "batch_size", "image_size", "image_type", "input_number"],

        "tensorflow_dataset": {
            "value": None,
            "desc": "tensorflow官方提供的数据集，若该值不设为None，则会启用tfds.load加载数据集。"
        },
        "data_dir": {
            "value": "../../data/",
            "desc": "训练数据集所在路径，若为相对位置，则相对于相对位置为当前应用app所在位置。若启用tfds，则会将目录下载在该目录下。"
        },
        "preprocess_function": {
            "value": "eidolon.loader.process_image_classification"
        },
        "features": {
            "value": [],
            "desc": "需要加载的数据字段"
        },
        "data_types": {
            "value": [],
            "desc": "需要加载的数据类型"
        },
        "buffer_size": {
            "value": 0,
            "desc": "(遗留字段)载入训练图像时设置的缓存区大小，若设置为0，则一次性载入所有图像。（在训练数据量不大的时候推荐设置为0）"
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
            "desc": "（遗留字段）图像格式，可选：jpg, png, bmp。"
        },
        "input_number": {
            "value": 1,
            "desc": "（遗留字段）单次输入图片数量，部分应用需要输入多张图片才能输出一张图片。默认输入一张。"
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
        "keys": ["epoch", "save_period", "checkpoints_dir", "max_image_save", "log_dir", "tensorboard_enable", "training_device", "remove_history_checkpoints", "load_latest_checkpoints"],
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
            "desc": "保存周期。每过一个保存周期，将会运行测试数据，保存训练的检查点，以及记录训练日志。"
        },
        "max_image_save": {
            "value": 5,
            "desc": "单轮保存图像限制，若超出部分，超出部分将不会保存。"
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
            "desc": "启用tensorboard记录日志。可以根据个人习惯接入tensorboard在页面查看训练情况。"
        },
        "training_device": {
            "value": "default",
            "desc": "训练使用设备，若需要指定设备，请完整指定设备编号，如：CPU:0，GPU:0。若使用默认设备，输入default。注意：tensorflow默认会占用所有显卡的内存，因此规定仅仅使用某一块显卡，请指定-only,如：GPU:7-only"
        },
        "remove_history_checkpoints": {
            "value": True,
            "desc": "只保留最近的训练检查点。若启用该设置，早期检查点均会被移除且无法复原。（注意：若禁用该设置，则所有保存点会被保留，会占用大量磁盘空间。）"
        },
        "load_latest_checkpoints": {
            "value": True,
            "desc": "载入最近的检查点。若关闭该设置，将从头开始训练。"
        },
        "lambda": {
            "value": [1, 1],
            "desc": "损失函数超参数。输入内容为数组，每个元素请用空格隔开，依次为每一项。"
        }
    },
    "pixel": {
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
            "desc": "遗留配置，仅对默认UNET有效，如果在低配GPU中(<4G)，可能发生网络结构过于复杂而显存不足的情况。禁用该选项时，会把UNet中的编码器最后一层与解码器第一层去除。"
        },
        "callback_args": {
            "value": [],
            "desc": "设置回调函数输入参数，用于初始化自定义的回调参数。目前支持--key=value格式。输入内容为数组，每个元素请用逗号隔开。"
        }
    },
    "classification": {
        "model_function": {
            "value": "eidolon.model.basic.make_Conv_models",
            "desc": "模型构造函数。"
        },
        "loss_function": {
            "value": "tensorflow.keras.losses.BinaryCrossentropy",
            "desc": "训练损失函数，可选:binary_cross_entropy, categorical_cross_entropy, sparse_categorical_crossentropy"
        },
        "labels": {
            "value": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            "desc": "分类标签名称"
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
            # runtime path
            self.runtime_path = self.config["runtime"]["path"]["value"]

            # data dir
            self.data_dir = self.config["dataset"]["data_dir"]["value"]
            # tfds
            self.tf_data = self.config["dataset"]["tensorflow_dataset"]["value"]
            # load function
            self.preprocess_function = self.config["dataset"]["preprocess_function"]["value"]
            self.data_features = self.config["dataset"]["features"]["value"]
            self.data_types = self.config["dataset"]["data_types"]["value"]
            # buffer size
            self.buffer_size = self.config["dataset"]["buffer_size"]["value"]
            # batch size
            self.batch_size = self.config["dataset"]["batch_size"]["value"]
            # image size
            self.image_size = self.config["dataset"]["image_size"]["value"]
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
            # max save images
            self.max_image_save = self.config["training"]["max_image_save"]["value"]
            # checkpoints
            self.checkpoints_dir = self.config["training"]["checkpoints_dir"]["value"]
            # log dir
            self.log_dir = self.config["training"]["log_dir"]["value"]
            # tensorboard enable
            self.tensorboard_enable = self.config["training"]["tensorboard_enable"]["value"]
            # training device
            self.training_device = self.config["training"]["training_device"]["value"]
            # remove history
            self.remove_history_checkpoints = self.config[
                "training"]["remove_history_checkpoints"]["value"]
            # load latest checkpoint
            self.load_latest_checkpoint = self.config["training"]["load_latest_checkpoints"]["value"]
            # parse lambda array
            self.lambda_array = self.config["training"]["lambda"]["value"]

            # pixel to pixel
            # generator
            self.generator = self.config["pixel"]["generator"]["value"]
            # discriminator
            self.discriminator = self.config["pixel"]["discriminator"]["value"]
            self.high_performance = self.config["pixel"]["high_performance"]["value"]
            # callback args
            self.callback_args = self.config["pixel"]["callback_args"]["value"]

            # classify
            self.classify_model_function = self.config["classification"]["model_function"]["value"]
            self.classify_loss_function = self.config["classification"]["loss_function"]["value"]
            self.classify_labels = self.config["classification"]["labels"]["value"]

    def get(self, belong, key):
        """获取某个参数

        Arguments:
            belong {[type]} -- [description]
            key {[type]} -- [description]

        Returns:
            [type] -- [description]
        """
        try:
            return self.config[belong][key]["value"]
        except Exception:
            return None


class ArgsParser:
    """
    将输入参数转化成key-vale,F
    主要用于回调函数的参数中，支持--key=value格式
    @author yuwei
    @since 2019.12.6
    """

    def __init__(self, args):
        """
        传入带解析的args数组
        """
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


# 若单独运行，则会创建一个配置模板
if __name__ == "__main__":

    # -d 保存路径，如果不设置，为当前路径
    # -r 运行时环境，如果不设置，则为空

    options, _ = getopt.getopt(sys.argv[1:], "d:r:", ["dir=", "runtime="])
    print(options)
    # 保存目录，默认位置为当前位置
    config_file = "./"
    runtime = None
    for key, value in options:
        if key in ("-d", "--dir"):
            config_file = value
        if key in ("-r", "--runtime"):
            runtime = value

    # 更新配置文件路径
    config_file = os.path.join(config_file, "config.json")
    # 添加运行时
    if runtime != None:
        default_config["runtime"]["path"]["value"].append(runtime)
        
    # 在指定位置创建配置文件
    create_config_JSON_temple(config_file)
