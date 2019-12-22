import tensorflow as tf

import train_tool
import loader

import os
import time
import importlib


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
            log_dir=config_loader.log_dir, save_period=config_loader.save_period)
        print("Initial train log....")

        # 初始化检查点路径
        self.checkpoint_prefix = os.path.join(
            self.config_loader.checkpoints_dir, "ckpt")

        #  待保存的模型
        self.model_map = {}

    def on_prepare(self):
        """
        准备阶段调用，允许重写。
        默认将执行以下功能：
        1. 若没有显示的创建检查点，将手动创建检查点
        """
        # 创建checkpoint
        self.checkpoint = tf.train.Checkpoint()
        # 创建映射
        self.checkpoint.mapped = self.model_map
        print("Initial checkpoint....")
        model_list_name = []
        for name in self.model_map:
            model_list_name.append(name)
        print("Models: {}".format(model_list_name))

        # 加载模型
        if self.config_loader.load_latest_checkpoint == True:
            #寻找最新的模型
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

    def on_train(self, current_epoch):
        """
        训练开始调用，允许重写
        默认完成以下任务：
        1. 每轮训练完，移除历史检查点。
        2. 每轮训练完，保存模型检查点。该函数默认不提供内容。
        """
        # saving (checkpoint) the model every 20 epochs
        if current_epoch % self.config_loader.save_period == 0:
            # remove history checkpoints
            if self.config_loader.remove_history_checkpoints == True:
                train_tool.remove_history_checkpoints(
                    self.config_loader.checkpoints_dir)
                print("Remove history checkpoints....")

            # save the checkpoint
            self.checkpoint.save(file_prefix=self.checkpoint_prefix)
            print("Store current checkpoint successfully....")

    def on_test(self, current_epoch):
        """
        测试开始调用，允许重写
        默认不提供任何方法。
        """
        pass

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

        # start training
        print("Start training, epochs={}".format(self.config_loader.epoch))
        # start from last time
        print("Start from epoch={}".format(self.log_tool.current_epoch))
        print("------------------------------------------------------------------\n")
        for epoch in range(self.log_tool.current_epoch, self.config_loader.epoch+1):
            # initial time
            start = time.time()

            # inovke each round
            #train epoch
            self.on_train(epoch)
            #test epoch
            self.on_test(epoch)

            # update epoch
            self.log_tool.update_epoch()
            print("Time taken for epoch {} is {} sec".format(
                epoch, time.time() - start))
            print("------------------------------------------------------------------\n")

        print("Total Training process finished....")
        print("Sweep up....")
        self.on_finish()
        print("Finished.")


def main(config_loader):
    """
    守护进程接口，daemon.py会调用该接口
    """

    # 创建训练类
    # 名称是以点的形式创建
    # split the module name and class name
    module_name, class_name = config_loader.container.split(".")
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
        with tf.device(config_loader.training_device):
            container.lifecycle()
