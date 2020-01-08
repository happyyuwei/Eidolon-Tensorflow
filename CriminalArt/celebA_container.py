from eidolon.train import Container
from CriminalArt import load_celebA
from CriminalArt import model
from CriminalArt import evaluate

import os

import tensorflow as tf


class CelebAContainer(Container):

    def __init__(self, config_loader):
        """
        调用父类初始化
        """
        super(CelebAContainer, self).__init__(config_loader)

    def on_prepare(self):

        # 载入数据集
        train_dataset = load_celebA.load_dataset(
            self.config_loader, is_training=True)
        test_dataset = load_celebA.load_dataset(
            self.config_loader, is_training=False)

        # 注册数据
        self.register_dataset(train_dataset, test_dataset)

        # 创建模型
        self.model = model.Resnet(
            [self.config_loader.image_height, self.config_loader.image_height, 3])

        print("Initial resnet50....")
        self.log_tool.plot_model(self.model, "resnet50")

        # 创建优化器
        self.optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.9)
        print("Initial generator optimizer....")

        # 将模型保存到存储列表，以便框架自动保存
        self.model_map["resnet"] = self.model
        self.optimize_map["resnet_optimizer"] = self.optimizer

        # 父类准备函数
        super(CelebAContainer, self).on_prepare()

    @tf.function
    def on_train_batch(self, input_image, label):
        # 传入的label包含label和mask
        label, _ = label
        with tf.GradientTape() as tape:
            # 输出
            logit = self.model(input_image)

            # 计算损失，多标签损失，将每个标签当做0-1,在softmax
            loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=logit, labels=label))

            # 记录损失和输出
            loss_set = {
                "loss": loss
            }

            # 梯度求解
            gradients = tape.gradient(loss, self.model.trainable_variables)
            # 优化网络参数
            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables))

        return loss_set

    def on_test_epoch(self, current_epoch, loss_set):
         # 保存损失与定量测试结果
        self.log_tool.save_loss(loss_set)

        # 测试可视化结果
        for test_input, test_target in self.test_dataset.take(1):

            test_target, _ = test_target
            # 生成测试结果
            predicted_label = self.model(test_input[0:1])

        # 生成特征的视觉图像
        predicted_visual_feature = evaluate.create_visual_tensor(
            predicted_label)
        target_visual_feature = evaluate.create_visual_tensor(test_target)

        image_list = [test_input, target_visual_feature,
                      predicted_visual_feature]
                      
        title_list = ["IN", "GT", "PR"]
        self.log_tool.save_image_list(image_list, title_list)
