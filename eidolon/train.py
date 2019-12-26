# third part lib
import tensorflow as tf

# inner lib
from eidolon import train_tool
from eidolon import loader

# system lib
import os
import time
import importlib
import datetime


class Container:
    """
    训练容器管理整个训练生命周期。
    需要注意的是：该类只提供基本模板，无法直接使用，需要继承该类进行具体实现。
    整个生命周期包括：on_prepare, on_train, on_finish.
    on_prepare:准备阶段，在执行完构造函数__init__()后将直接调用
    on_train: 训练阶段，该函数在每一轮迭代开始时调用。
    on_finish: 结尾阶段，该函数在训练完成之后调用。
    """

    def __init__(self, config_loader):

        # 保存配置
        self.config_loader = config_loader

        # 初始化日志工具
        self.log_tool = train_tool.LogTool(
            log_dir=config_loader.log_dir, save_period=config_loader.save_period, tensorboard_enable=config_loader.tensorboard_enable)
        print("Initial train log....")

        # 初始化检查点路径
        self.checkpoint_prefix = os.path.join(
            self.config_loader.checkpoints_dir, "ckpt")

        #  待保存的模型， 继承方法需要保存模型
        self.model_map = {}
        # 优化器，继承方法需要注册所有优化器
        self.optimize_map = {}
        # 数据集, 继承方法需要注册数据集
        self.train_dataset = None
        self.test_dataset = None

    def register_dataset(self, train_dataset, test_dataset):
        """
        注册数据集，供训练
        """
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def on_prepare(self):
        """
        准备阶段调用，允许重写。
        默认将执行以下功能：
        1. 若没有显示的创建检查点，将手动创建检查点
        """
        # 创建checkpoint
        self.checkpoint = tf.train.Checkpoint()

        # 创建映射，将model_map和optimze_map整合成一个map，使得chekcpoint能够将其全部保存
        ckpt_map = {}
        ckpt_map.update(self.model_map)
        ckpt_map.update(self.optimize_map)

        # 设置映射至checkpoint
        self.checkpoint.mapped = ckpt_map
        print("Initial checkpoint....")
        model_list_name = []
        for name in ckpt_map:
            model_list_name.append(name)
        print("Checkpoints: {}".format(model_list_name))

        # 加载模型
        if self.config_loader.load_latest_checkpoint == True:
            # 寻找最新的模型
            checkpoint_states = tf.train.get_checkpoint_state(
                self.config_loader.checkpoints_dir)
            # 如果模型不存在，则启动新训练
            if checkpoint_states != None:
                latest_checkpoint = checkpoint_states.model_checkpoint_path
                self.checkpoint.restore(latest_checkpoint)
                print("Load latest checkpoint: {}....".format(latest_checkpoint))
            else:
                print("No checkpoints found in {}. Training from beginning.".format(
                    self.config_loader.checkpoints_dir))

    @tf.function
    def on_train_batch(self, input_image, target):
        """
        每一批的损失, 该函数需要返回损失函数结果的字典。
        该方法默认不提供内容
        """
        pass

    def on_train_epoch(self, current_epoch):
        """
        每一轮训练调用，允许重写
        默认完成以下任务：
        1. 获取数据集体
        2. 调用on_train_batch计算每一批的损失
        """
        """
        重写训练父类方法
        """
        for image_num, (input_image, target) in self.train_dataset.enumerate():
             # @since 2019.11.28 将该打印流变成原地刷新
            print("\r"+"input_image {}...".format(image_num), end="", flush=True)
            # 训练一个batch
            loss_set = self.on_train_batch(input_image, target)
        # change line
        print()
        # 返回 loss结果集
        return loss_set

    def on_save_epoch(self, current_epoch):
        """
        每个保存周期会被调用
        """
        # saving (checkpoint) the model every 20 epochs
        # remove history checkpoints
        if self.config_loader.remove_history_checkpoints == True:
            train_tool.remove_history_checkpoints(
                self.config_loader.checkpoints_dir)
            print("Remove history checkpoints....")

        # save the checkpoint
        self.checkpoint.save(file_prefix=self.checkpoint_prefix)
        print("Store current checkpoint successfully....")

        # 将最终模型进行保存成h5文件格式，只保存最后一个
        # 目前只支持固定路径，无法配置
        path = "./model"
        if os.path.exists(path) == False:
             # 重新创建文件夹
            os.mkdir(path)

        # 保存所有模型
        for name in self.model_map:
            model_path = os.path.join(path, "{}.h5".format(name))
            self.model_map[name].save(
                model_path, overwrite=True, include_optimizer=False)
            print("save {} model in HDFS file.".format(name))

        print("----------------------------------------------------------------------------------\n")

    def on_test_epoch(self, current_epoch, loss_set):
        """
        测试开始调用，允许重写
        默认不提供任何方法。
        """
        pass

    def on_train(self):
        """
        训练开始调用，允许重写
        默认完成以下任务：
        1. 管理每一轮训练，每一轮训练调用on_train_epoch()
        2. 每轮训练完，调用on_test_epoch()进行测试。
        3. 每轮训练完，调用on_save_epoch()保存模型检查点。
        """
        # start training
        print("Start training, epochs={}".format(self.config_loader.epoch))
        # start from last time
        print("Start from epoch={}".format(self.log_tool.current_epoch))
        for epoch in range(self.log_tool.current_epoch, self.config_loader.epoch+1):
            print("------------------------------------------------------------------\n")
            #reocrd current time
            print("start epoch {} at time: {}".format(
                epoch, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            # initial time
            start = time.time()

            # inovke each round
            # train epoch
            loss_set = self.on_train_epoch(epoch)

            # 没过一个保存阶段就会调用一次该函数
            if epoch % self.config_loader.save_period == 0:
                # test epoch
                self.on_test_epoch(epoch, loss_set)

            # update epoch
            self.log_tool.update_epoch()
            print("Time taken for epoch {} is {} sec".format(
                epoch, time.time() - start))
            print("------------------------------------------------------------------\n")
            # 保存，每个周期保存一次
            if epoch % self.config_loader.save_period == 0:
                self.on_save_epoch(epoch)

    def on_finish(self):
        """
        完成阶段调用，允许重写
        默认完成以下任务
        1. 保存最终模型
        """

        pass

    def lifecycle(self):
        """
        训练管理，负责训练， 记录每轮情况
        该代码块有原先训练代码抽离
        @since 2019.11.20
        @author yuwei
        """
        print("Start prepare....")
        self.on_prepare()

        print("Start training....")
        self.on_train()

        print("Total training process finished....")
        print("Sweep up....")
        self.on_finish()
        print("Finished.")


def main(config_loader):
    """
    守护进程接口，daemon.py会调用该接口
    """

    # 创建训练类
    # 名称是以点的形式创建，如：eidolon.pixel_container.PixelContainer
    # split the module name and class name
    split_list = config_loader.container.split(".")

    module_name = ".".join(split_list[0:len(split_list)-1])
    class_name = split_list[-1]
    print("load container：model_name={}, class_name={}".format(
        module_name, class_name))

    # load training function in train_watermark
    o = importlib.import_module(module_name)
    Container = getattr(o, class_name)
    container = Container(config_loader=config_loader)

    if config_loader.training_device == "default":
        print("Use default device....")
        # 启动生命周期
        container.lifecycle()
    else:
        # 在指定设备中运行
        print("Use device: {} ....".format(config_loader.training_device))

        #若允许只是用一块GPU内存
        if config_loader.training_device.endswith("-only"):
            print("This program will only use device GPU:{}....".format(device_id))
            # parse device id
            device_id=config_loader.training_device.split("-")[0].split(":")[1]
            # 把其他显卡设成不可见
            os.environ["CUDA_VISIBLE_DEVICES"]=device_id

            #启动生命周期
            container.lifecycle()
        else:
            print("This program will only use all the GPU Menmory....")
            # 默认占用所有GPU内存
            with tf.device(config_loader.training_device):
                container.lifecycle()
