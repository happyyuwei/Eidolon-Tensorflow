import tensorflow as tf
import numpy as np


def evaluate(image1, image2, psnr_enable=True, ssim_enable=True, ber_enable=False):
    """
    this is for eval
    可测试psnr, ssim, ber
    :param: image1: tensor structure with [..,128,128,3] or [..,32,32,3], et. al, the scale is [-1,1]
    :param: image2: tensor structure with [..,128,128,3] or [..,32,32,3], et. al, the scale is [-1,1]
    :param: ssim_enable
    :param: psnr_enable
    :param: ber_enable
    :return result_set, it is a dir
    """
    # 将尺寸变为[0，1]
    image1 = image1*0.5+0.5
    image2 = image2*0.5+0.5

    # result
    result_set = {}

    # invoke
    if psnr_enable == True:
        psnr = tf.image.psnr(image1, image2, max_val=1)
        result_set["psnr"] = psnr

    if ssim_enable == True:
        ssim = tf.image.ssim(image1, image2, max_val=1)
        result_set["ssim"] = ssim

    if ber_enable == True:
        image1 = np.array(image1[0])
        image2 = np.array(image2[0])
        # filter
        image1[image1 >= 0.5] = 1
        image1[image1 < 0.5] = 0
        image2[image2 >= 0.5] = 1
        image2[image2 < 0.5] = 0
        # ber
        ber = np.mean(np.abs(image1-image2))
        # 先就这么写，把tensor值转成数字
        result_set["ber"] = ber

    # results
    return result_set


