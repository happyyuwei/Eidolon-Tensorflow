"""
本实验会对比嵌水印和不嵌水印的PSNR, SSIM。分别对比不嵌水印，嵌入二值水印，嵌入彩色水印。
"""

import getopt
import os
import pickle
import sys
# 该目录为根目录
sys.path.append("../../../")
from WMNetv2.eval import eval_all


if __name__ == "__main__":

    # params
    dataset = None
    model = None
    watermark = None
    is_binary = None
    visual_result = None
    wm_width = None

    # parse input arguments if exist
    if len(sys.argv) >= 2:
        options, _ = getopt.getopt(sys.argv[1:], "", [
                                   "data=", "model=", "watermark=", "is_binary=", "visual_result=", "wm_width="])
        print(options)

        for key, value in options:
            if key in ("--data"):
                dataset = value
            if key in ("--model"):
                model = value
            if key in ("--watermark"):
                watermark = value
            if key in ("--wm_width"):
                wm_width = int(value)
            if key in ("--is_binary"):
                is_binary = (value in ("true", "True"))
            if key in ("--visual_result"):
                visual_result = value

    if model == None:
        print("Error: No model found...")

    if dataset == None:
        print("Error: No dataset found...")

    print("\nstart {} testing---------------------------------------------------------------".format(model))
    # evaluate model with watermark
    results, report = eval_all(data_path=dataset,
                               model_path=model,
                               visual_result_dir=visual_result,
                               watermark_path=watermark,
                               watermark_binary=is_binary,
                               wm_width=wm_width,
                               wm_height=wm_width)

    print("{}---------------------------------------------------------------------------------------".format(model))
    print(report)
    # it seemed not used
    # f = open('./{}/result_wm.pkl'.format(dataset), 'wb')
    # pickle.dump(results, f)
    # f.close()
