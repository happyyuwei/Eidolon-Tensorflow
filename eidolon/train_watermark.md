# 如何给网络添加水印


## 准备
本项目底层基于 `Tensorflow` 实现，需要安装 Tensorflow 2.0 或以上版本。
稳定测试平台：
 版本|windows 10 | Centos | Ubuntu
---|---|---|---|
`tensorflow 2.0`|Yes|Yes|Yes 
`tensorflow 2.1`|Yes|Yes|Yes
`tensorflow 1.x`|No|No|No
高于 `2.1`|-|-|-

* 当时开发与运行使用 `tensorflow2.0/2.1`，在`window/Linux`均有很好支持。不确定是否完全支持后续版本，理论上应该不成问题，`TF`在2.0以后兼容性变好了。
* 不支持 `tensorflow 1.x`
* `Python 3.x`，不支持 `Python 2.x`
* `CPU/GPU` 平台均可，只是`CPU`速度慢

## 更多准备
本训练使用作者自己曾经造的蹩脚轮子 `Tensorflow-Eidolon`，在 `tensorflow` 计算层之上进行一些训练生命周期的管理。详情： https://github.com/happyyuwei/Eidolon-Tensorflow。这上面的版本比本训练用到的版本更新一点。


## 开始
切换到工程目录，给你发的代码结构里，在./codes/WMNetv2下面。

WMNetv1不要管了。

执行创建工程：
```js
python create.py --name=watermark_add
```
`watermark_add`可以换其他任何自己取的工程名，这个工程将承载整个训练生命周期。

此时，当前目录下的 `app/`目录下会出现一个`watermark_add`或者你取的其他名字的文件夹。

里面包括四个文件，
文件名| 描述
---|---|
config.bat| 历史遗留，可以删。
config.json |重点关注，所有训练的配置文件，里面的每一项都跟着解释
paint_loss.bat/paint_loss.sh| 在没有接入tensorboard时，用于简易画出损失曲线，脚本后缀取决于系统。windows环境，`.bat`, linux环境，`.sh`
train.bat/train.sh| 训练入口，运行该脚本开始训练。

## 准备数据集

请自行下载，任何图像变换对（`x->y`）都可以。比如经典论文：https://arxiv.org/abs/1611.07004，或者去本论文的实验参考文献中寻找。这步应该是最花时间的。拿到数据集，请按以下步骤存放：
* 把每一对图片（`x，y`）拼成一张图，左边标签，右边输入。图片命名随意。
* 新建一个文件夹，取名：dataset 或者其任何名字
* 里面建两个文件夹，train 和 test。从功能上讲，test文件夹是validation，当时设计之初还不懂，后续改名就麻烦了。
* 把训练集放到train文件夹里，验证集放到test文件夹里。
* 完成

## 修改配置文件
大部分可以默认，或者你自己定制，但注意几个地方

* `data_dir` 设置成你刚才数据集的路径
* `batch_size` 请按照你机器的计算能力自行修改
* `image_size` 和 `crop_size` 数据集原图会变换成 `image_size+crop_size` 的大小, 然后从中随机裁剪`image_size`大小进行训练。如果不想这样的数据增强，请设`crop_size=0`
* `save_period` 自行调整。每个保存周期会计算训练集和测试集的损失，同时会把`checkpoint`和`keras`中支持的`.h5` 模型（忘了这个版本有没有保存h5）保存下来，若网络很大，每轮保存会很占存储空间。
* `tensorboard_enable`，若习惯使用 `tensorboard`，可以启动。启动后无需任何其他配置，打开网页即可查看训练曲线。
* `training_device` 训练设备，若是多块GPU同时训练，默认会占用所有资源，如果仅仅想在部分显卡上，请设置 `GPU:{显卡编号}-only` 如`GPU:0-only`。如果单cpu/gpu则无需顾虑。

## 开始训练
运行当前工程下的train.bat/train.sh
如果想停止，直接关掉，下一次训练会从上一个保存周期开始

结束

## 查看效果

* 每一轮的训练数据在，`{当前工程}/log/train_log.txt`下，运行./paint_loss.bat 看可视化曲线。在linux下好像是把曲线保存成图片在当前目录下，不记得了。
* 图片输出结果在 `{当前工程}/log/result_image`下
* 所有`checkpoint`在`{当前工程}/training_checkpoint`下
* 若使用 tensorboard，上文说需要在配置文件里启用。剩余和官方教程里一样。输入下方指令即可查看
```
tensorboard --logdir={当前工程}/log/tensorboard
```

https://github.com/happyyuwei/Eidolon-Tensorflow 有一些例子可以查看。


## 水印在哪里
到目前其实还没加水印，只是训练普通的 pixel2pixel，如果能成功运行，说明框架没问题。

现在，可以新建一个项目，重做上面的步骤；或者直接使用这个工程，把checkpoint和log都删了。

在运行前，要多做两件事：
* 把配置文件中，`container`设为`WMNetv2.wm_container.WMContainer`。这个是已实现的给网络加水印的容器
* 准备一张水印图片，callback_args里把--wm_path改成你的水印，或者用上面默认的也可以。
* `wm_width` 是水印大小，可自行修改。

一样运行，
一样等待结果。












