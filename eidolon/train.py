# third part lib
import tensorflow as tf

# inner lib
from eidolon import train_tool
from eidolon import loader

# system lib
import os
import sys
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
        # 优化器名称列表，记录所有优化器名称，用于创建梯度带的时候加速读取
        self.optimizer_name_list = []
        # 模型与优化器名称的对应关系
        self.model_optimzer_map = {}
        # 数据集, 继承方法需要注册数据集
        self.train_dataset = None
        self.test_dataset = None


        # 损失记录器列表
        self.loss_recorder_map={}

    def register_dataset(self, train_dataset, test_dataset=None):
        """
        注册数据集，供训练
        """
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def register_model_and_optimizer(self, optimizer,  model_map, optimizer_name):
        """
        :param: optimizer 优化器实例
        :param: optimizer_name 该优化器实例名称
        :param: model_map 该优化器对应的模型
        注册模型与优化器，允许反复调用，每次调用会将该传入的参数进行添加。
        指定模型与优化器对应关系
        @since 2020.4.9
        @author yuwei
        """
        # 注册优化器
        self.optimize_map[optimizer_name] = optimizer
        # 注册模型
        self.model_map.update(model_map)
        # 添加对应关系
        self.model_optimzer_map[optimizer_name] = model_map
        # 添加名称
        self.optimizer_name_list.append(optimizer_name)
    
    def register_display_metrics(self, name_list):
        for each in name_list:
            #创建损失函数记录器
            self.loss_recorder_map[each]=tf.keras.metrics.Mean(name=each)

    def on_prepare(self):
        """
        准备阶段调用，允许重写。
        默认将执行以下功能：
        1. 若没有显示的创建检查点，将手动创建检查点
        2. 若需要加载检查点，则加载指定检查点
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

        # 输出模型列表在控制台
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

    def before_train_batch(self):
        """
        每轮训练时的准备.
        在on_train_batch前调用.
        返回的内容会传入on_train_batch的第二个参数中。
        @since 2019.2.19
        @author yuwei
        该函数引入是为了解决部分操作无法在使用@tf.function加速的代码段中进行。
        默认函数返回空
        """
        return "None"

    def compute_loss_function(self, each_batch, extra_batch_data):
        """
        返回loss_map和display_map
        loss_map用于训练损失，其key必须使用optimizer_name_list中的名字，value为loss值
        display_map用于可视化损失
        警告：不推荐在此处添加@tf.function
        """
        return {}, {}

    # @tf.function
    def gradient_tape_container(self, each_batch, extra_batch_data, optimizer_num, tape_map):
        """
        梯度容器,三个以上优化器可使用该函数。
        目前存在一些不稳定性。
        若使用@tf.function后可能存在运行完一批后出现卡住的问题，因此不推荐使用该函数。
        该函数同样可以用于一个与两个优化器，但是有专门的优化的函数，因此不推荐使用。
        """
        # 创建tape
        with tf.GradientTape() as tape:
            if optimizer_num < len(self.optimizer_name_list):
                # 保存当前梯度带
                tape_map[self.optimizer_name_list[optimizer_num]] = tape
                # 递归深度创建梯度带
                self.gradient_tape_container(each_batch, extra_batch_data, optimizer_num+1, tape_map)
            else:
                # 当所有梯度均已创建，则计算梯度信息，并返回结果
                loss_map, display_map = self.compute_loss_function(
                    each_batch, extra_batch_data)

                # 所有模型按照指定的优化器分别训练
                for optimizer_name in self.optimizer_name_list:

                    # 获取对应模型列表
                    model_map = self.model_optimzer_map[optimizer_name]
                    # print(model_map)

                    # 获取所有训练参数
                    trainable_variables = []
                    for each in model_map.values():
                        trainable_variables = trainable_variables+each.trainable_variables

                    # 計算梯度
                    gradients = tape_map[optimizer_name].gradient(
                        loss_map[optimizer_name], trainable_variables)

                    # 优化网络参数
                    self.optimize_map[optimizer_name].apply_gradients(
                        zip(gradients, trainable_variables))

                    for each in display_map:
                        self.loss_recorder_map[each](display_map[each])



    @tf.function
    def single_gradient_tape(self, each_batch, extra_batch_data):
        """
        计算梯度与损失函数优化。
        单个优化器的情况下，使用该函数。
        不建议覆盖该函数
        警告：覆盖发方法后推荐添加@tf.function注解，经测试可以节省40%训练时间。
        """
         # 由于只有一个优化器，因此index为0。
        optimizer_name = self.optimizer_name_list[0]
        # 获取模型列表，由于只有一个优化器，因此index为0。
        model_map = self.model_optimzer_map[self.optimizer_name_list[0]]
        # 获取所有训练参数
        trainable_variables = []
        for each in model_map.values():
            trainable_variables = trainable_variables+each.trainable_variables

        #获取优化器
        optimizer=self.optimize_map[optimizer_name]

        with tf.GradientTape() as tape:
           
            # 当所有梯度均已创建，则计算梯度信息，并返回结果
            loss_map, display_map = self.compute_loss_function(
                each_batch, extra_batch_data)
            
            # 計算梯度
            gradients = tape.gradient(
                loss_map[optimizer_name], trainable_variables)

            # 优化网络参数
            optimizer.apply_gradients(zip(gradients, trainable_variables))

            for each in display_map:
                self.loss_recorder_map[each](display_map[each])



    @tf.function
    def double_gradient_tape(self, each_batch, extra_batch_data):
        """
        计算梯度与损失函数优化。
        两个优化器的情况下，使用该函数。
        不建议覆盖该函数
        警告：覆盖发方法后推荐添加@tf.function注解，经测试可以节省40%训练时间。
        """
         # 由于只有两个优化器，因此index为0，1。
        optimizer1_name = self.optimizer_name_list[0]
        optimizer2_name = self.optimizer_name_list[1]

         # 获取模型列表，第一个优化器，index为0。
        model1_map = self.model_optimzer_map[self.optimizer_name_list[0]]
        # 获取所有训练参数
        trainable_variables1 = []
        for each in model1_map.values():
            trainable_variables1 = trainable_variables1+each.trainable_variables

        # 获取模型列表，第一个优化器，index为0。
        model2_map = self.model_optimzer_map[self.optimizer_name_list[1]]
        # 获取所有训练参数
        trainable_variables2 = []
        for each in model2_map.values():
            trainable_variables2 = trainable_variables2+each.trainable_variables

        optimizer1=self.optimize_map[optimizer1_name]
        optimizer2=self.optimize_map[optimizer2_name]

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
        
            # 当所有梯度均已创建，则计算梯度信息，并返回结果
            loss_map, display_map = self.compute_loss_function(
                each_batch, extra_batch_data)
        
            # 計算梯度
            gradients1 = tape1.gradient(
                loss_map[optimizer1_name], trainable_variables1)
            gradients2 = tape2.gradient(
                loss_map[optimizer2_name], trainable_variables2)

            # 优化网络参数
            optimizer1.apply_gradients(zip(gradients1, trainable_variables1))

            optimizer2.apply_gradients(zip(gradients2, trainable_variables2))

            for each in display_map:
                self.loss_recorder_map[each](display_map[each])





    def on_train_batch(self, each_batch, extra_batch_data):
        """
        :param: each_batch 训练数据
        :param: extra_batch_data before_train_batch()阶段读取的额外数据
        每一批的损失, 该函数需要返回损失函数结果的字典。
        该方法提供默认训练内容，根据准备阶段绑定的对应关系，将模型指定至对应优化器。
        警告：覆盖发方法后推荐添加@tf.function注解，经测试可以节省40%训练时间。
        """
        # 创建梯度带
        # 计算梯度并更新参数
        # 返回需要记录与展示的结果集
        if len(self.optimizer_name_list)==1:
            #调用单梯度计算
            self.single_gradient_tape(each_batch,extra_batch_data)
        elif len(self.optimizer_name_list)==2:
            # 调用双梯度计算
            self.double_gradient_tape(each_batch,extra_batch_data)
        else:
            # 调用梯度容器
            self.gradient_tape_container(each_batch, extra_batch_data, 0, {})

    def get_display_map(self):
        """
        在添加@tf.function后，无法在更新参数时获取到loss，因此目前方案为在随机采样一批数据，计算其当前损失。
        @author yuwei
        @since 2020.4.11
        """
        #采样数据集
        sample_batch=self.train_dataset.take(1)
        #额外数据
        extra_batch=self.before_train_batch()
        #计算损失
        for each in sample_batch:
            _, display_map=self.compute_loss_function(each, extra_batch)

        return display_map


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
        #重置损失函数记录器
        for reocrder in self.loss_recorder_map.values():
            reocrder.reset_states()
        #开始训练
        for image_num, each_batch in self.train_dataset.enumerate():
            # @since 2019.11.28 将该打印流变成原地刷新
            print("\r"+"input_batch {}...".format(image_num), end="", flush=True)
            # 准备一个batch
            extra_batch_data = self.before_train_batch()
            # 训练一个batch
            self.on_train_batch(each_batch, extra_batch_data)

        print()
        #计算loss结果集
        # 返回 loss结果集
        # return self.get_display_map()

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

    def compute_test_metrics_function(self, each_batch, extra_batch_data):
        """
        测试函数，默认不提供任何实现
        返回key-value键值对，其中key为评价指标名称
        """
        return {}
        

    def on_test_metrics(self):
        """
        所有定量数据评估内容全部在该方法中实现。
        返回结果集，为key-value结构，其中key为名称，value为数字，会与display_map一起合并展示。
        默认不提供方法，需要父类覆盖。
        """
        
        if self.test_dataset!=None:
            display_map={}
            count=0
            for each_batch in self.test_dataset:
                extra_batch_data=self.before_train_batch
                metrics_map=self.compute_test_metrics_function(each_batch, extra_batch_data)

                #第一次计算
                if len(display_map)==0:
                    display_map=metrics_map
                else:
                    #计算平均值
                    for each in display_map:
                        display_map[each]=display_map[each]+metrics_map[each]
                
                count=count+1
                
            #计算平均值
            for each in display_map:
                display_map[each]=display_map[each]/count
            
            return display_map
        else:
            return {}

    def on_test_visual(self):
        """
        所有可视化评估内容全部在该方法中实现。
        返回结果集，为两个列表结构，其中第一个列表是图像集，第二个列表是图像名称。会进行可视化保存与展示。
        默认不提供方法，需要父类覆盖。
        """
        return [], []

    def on_test_epoch(self, current_epoch, display_map):
        """
        测试开始调用，允许重写。
        调用on_test_visual()与on_test_metrics()方法。
        保存与展示评估数据
        """
        # 进行定量数据评估
        data_map = self.on_test_metrics()
        # 进行可视化评估
        image_list, title_list = self.on_test_visual()

        display_map.update(data_map)
        # 保存损失与定量测试结果
        self.log_tool.save_loss(display_map)

        # 检查输出长度是否一致
        if len(image_list) != len(title_list):
            print("error: the lenght of image list and title list are not the same.")
            sys.exit(0)

        # 当需要保存可视化
        if len(image_list) > 0:
            # 保存可视结果
            self.log_tool.save_image_list(image_list, title_list)

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
            # reocrd current time
            print("start epoch {} at time: {}".format(
                epoch, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            # initial time
            start = time.time()

            # inovke each round
            # train epoch
            self.on_train_epoch(epoch)

            # 获取所有展示值
            display_map={}
            for each in self.loss_recorder_map:
                display_map[each]=self.loss_recorder_map[each].result()

            # 每过一个保存阶段就会调用一次该函数
            if epoch % self.config_loader.save_period == 0:
                # test epoch
                self.on_test_epoch(epoch, display_map)

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

        # 若允许只是用一块GPU内存,
        # Todo 问题尚未解决，一直都会使用所有内存
        if config_loader.training_device.endswith("-only"):
            # parse device id
            device_id = config_loader.training_device.split(
                "-")[0].split(":")[1]
            print(
                "This program will only use the memory of device GPU:{}....".format(device_id))
            # 把其他显卡设成不可见
            os.environ["CUDA_VISIBLE_DEVICES"] = device_id

            # 启动生命周期
            container.lifecycle()
        else:
            print("This program will use all the GPU Menmory....")
            # 默认占用所有GPU内存
            with tf.device(config_loader.training_device):
                container.lifecycle()
