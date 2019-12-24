import os
import sys
import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt

sys.path.append("./")

from WMNetv2 import auto


class Model:

    def __init__(self, model_path):
        """
        检查模型是否存在，非常重要！！
        tensorflow的checkpoint不会检查,若路径输入错误不会报错！！！
        大坑啊，只能手动实现。建议所有模型类继承该基类
        @author yuwei
        @since 2019.11.27
        """
        if os.path.exists(model_path) == False:
            print("Error: No model is found in {}".format(model_path))
            sys.exit()


class EncoderDecoder(Model):

    def __init__(self, model_path, key_enable=False):
        super(EncoderDecoder, self).__init__(model_path)

        self.key_enable=key_enable

        print("load encoder....")
        self.encoder = auto.Encoder()
        print("load decoder....")
        self.decoder = auto.Decoder()

        if key_enable == True:
            print("load key encoder....")
            self.key_encoder = auto.Encoder()

        # initial optimizer
        print("initial optimizer....")
        # optimzer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        checkpoint = tf.train.Checkpoint()
        model_map={
            # "optimzer":optimzer,
            "encoder":self.encoder,
            "decoder":self.decoder
        }

    
        # load trained model
        if key_enable == True:
            #
            pass

        checkpoint.mapped = model_map
        print("load model....")
        checkpoint.restore(tf.train.latest_checkpoint(model_path))

    def decode_from_image(self, encode_image_path, secret_key_path=None):

        # read the feature [0,1]
        encode_image = plt.imread(encode_image_path)
        # change scale to [-1,1]
        encode_image = encode_image[:, :, 0:3] * 2 - 1

        # reshape to 4 dim
        encode_image = tf.reshape(encode_image, [1, 128, 128, 3])

        secret_key=None
        if self.key_enable==True:
            #检查是否输入秘钥
            if secret_key_path==None:
                print("Error: no secret key path found")
                sys.exit()
            # 载入秘钥
            secret_key = np.load(secret_key_path)
            # convert to tensor
            secret_key = tf.convert_to_tensor(secret_key, dtype=tf.float32)

        # out
        out=self.decode(encode_image, secret_key)

        # back to[0,1] with size [32,32,3]
        out = out[0] * 0.5 + 0.5
        # the output is numpy
        return np.array(out)

    def decode(self, encode_image, secret_key=None):
        """
        解码内核，输入输出均为tensor 结构， 范围从【-1，1】，不能直接使用
        the input of encode_image must be tensorflow type, with size [1,128,128,3]
        the out structure is still tensorflow structure [1,32,32,3] with scale [-1,1]
        @since 2019.11.23
        @author yuwei

        @update 2019.11.29
        将秘钥引入进来，只有在创建对象时将key_enable设为true，参数secret_key 才有效， 该参数为tensorflow【-1,1】

        """
        # the encode_image is [1, 128,128,3], reshape to [1, 32,32,48]
        encode_image = tf.reshape(encode_image, [1, 32, 32, 48])

        if self.key_enable == True:
             # 秘钥特征
            key_output = self.key_encoder(secret_key, training=True)
            # 整合
            encode_image = tf.concat([encode_image, key_output], axis=-1)

        # decode, the output is [-1,1]
        out = self.decoder(encode_image, training=True)

        # the out structure is still tensorflow structure
        # [1,32,32,3]
        return out

    def encode(self, input_path):
        """
        编码图片
        """
        # read the feature [0,1]
        encode_image = plt.imread(input_path)
        # change scale to [-1,1], only get 3 channels
        encode_image = encode_image[:, :, 0:3] * 2 - 1

        # the input should be [1,32,32,1]
        # reshape to 4 dim
        encode_image = tf.reshape(encode_image, [1, 32, 32, 3])
        # encode, shape [1,32,32,48]
        encode_image = self.encoder(encode_image, training=True)
        # reshape to [1,128,128,3]
        encode_image = tf.reshape(encode_image, [1, 128, 128, 3])
        # back to [0,1]
        encode_image = encode_image[0]*0.5+0.5
        # return numpy
        return np.array(encode_image)


class GeneratorModel(Model):
    """
    生成模型
    """

    def __init__(self, model_path, watermark=False, decoder_path=None):
        # parent init function
        super(GeneratorModel, self).__init__(model_path)

        self.watermark = watermark

        print("load generator....")
        self.generator = cgan.Generator()

        if watermark == True:
            print("load extractor....")
            self.extractor = invisible_extract.ExtractInvisible()
            print("initial auto encoder-decoder....")
            self.decoder_model = EncoderDecoder(decoder_path)

        # initial optimizer
        # not used
        print("initial generator optimizer....")
        generator_optimizer = tf.train.AdamOptimizer(2e-4, beta1=0.5)

        # load trained model
        checkpoint = None
        if watermark == False:
            checkpoint = tf.train.Checkpoint(
                generator_optimizer=generator_optimizer, generator=self.generator)
        else:
            checkpoint = tf.train.Checkpoint(
                generator_optimizer=generator_optimizer, generator=self.generator, extractor=self.extractor)

        # load model
        checkpoint.restore(tf.train.latest_checkpoint(model_path))
        print("load models....")

    def generate(self, input_image, attack_test_func=None):
        """
        This is used after the model is trained
        the outputs are tensors.
        This is used to do testing
        @author yuwei
        @since 2019.11.25
        :input_image 输入图像是tensor结构[1,128,128,3], scale: [-1,1]
        :attack_test_func 在输入图像之后加入干扰进行抗攻击测试， @since2019.11.27
        """
        # input
        output_tensor = self.generator(input_image, training=True)
        wm_tensor = None
        wm_feature = None
        if self.watermark == True:
            # attack test
            if attack_test_func != None:
                output_tensor = attack_test_func(output_tensor)
            # detect watermark feature
            wm_feature = self.extractor(output_tensor, training=True)

            # the feature size is [1,128,128,3]
            # decode watermark
            wm_tensor = self.decoder_model.decode(wm_feature)
        # the out structure is still tensorflow structure [1,32,32,3]
        return output_tensor, wm_tensor, wm_feature

    def generate_image(self, input_path):
        """
        generate image
        """
        # load image
        input_image = plt.imread(input_path)
        # change scale to [-1,1]
        input_image = input_image[:, :, 0:3]*2-1
        # change to tensor
        input_image = tf.reshape(input_image, [1, 128, 128, 3])

        output_tensor, wm_tensor, _ = self.generate(input_image)
        # change tensor to image
        # resize to [128,128,3], change scale to [0,1]
        output_image = np.array(output_tensor[0, :, :, :]*0.5+0.5)

        if wm_tensor != None:
            wm_tensor = np.array(wm_tensor[0, :, :, :]*0.5+0.5)

        return output_image, wm_tensor


# testing
if __name__ == "__main__":

    # model = GeneratorModel(model_path="./trained_models/flower_unet_l1_bw",
    #                        watermark=True, decoder_path="./trained_models/auto_mnist")
    # o, w = model.generate_image(input_path="1_GT.png")
    # plt.figure(1)
    # plt.imshow(o)
    # plt.figure(2)
    # plt.imshow(w)
    # plt.show()

    # # load image
    # input_image = plt.imread("1_GT.png")
    # # change scale to [-1,1]
    # input_image = input_image[:, :, 0:3]*2-1
    # # change to tensor
    # input_image = tf.reshape(input_image, [1, 128, 128, 3])

    # create empty config, this is used when loading data
    # import config
    # import loader
    # configs = config.ConfigLoader()
    # # load data
    # data_loader = loader.DataLoader("./data/flower/train", is_training=False)
    # dataset = data_loader.load(configs)

    # for input_image, ground in dataset.take(1):
    #     o, w, wf = model.generate(input_image)
    #     w = np.array(w[0, :, :, :]*0.5+0.5)
    #     plt.imshow(w)
    #     plt.show()
        # import train_tool
        # train_tool.save_images(image_list=[o,w,wf], title_list=["A","B","C"], image_dir="./")

    # start model
    # model = EncoderDecoder("./trained_models/auto_mnist")
    model = EncoderDecoder("./trained_models/auto_mnist", key_enable=False)

    # # get feature
    # wm_feature = model.encode("wm_binary.png")

    # # save feature
    # plt.imsave("wm_binary_feature.png", wm_feature)

    # decode feature
    wm = model.decode_from_image("./WMNetv2/watermark/2.png")

    # f=model.encode("./WMNetv2/watermark/wm_binary.png")  

    wm[wm < 0.5] = 0
    wm[wm > 0.5] = 1
    # print(wm.shape)
    # show watermark
    plt.imshow(wm)
    plt.show()
    # plt.imsave("x.png", f)
