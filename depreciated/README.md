# PaddlePaddle论文复现杂谈

## 背景


注：该repo基于PaddlePaddle，对AlexNet进行复现。时间仓促，难免有所疏漏，如果问题或者想法，欢迎随时提issue一块交流。

飞桨论文复现赛地址：[https://aistudio.baidu.com/aistudio/competition/detail/106](https://aistudio.baidu.com/aistudio/competition/detail/106)

不可多得的学习过程中还能搞钱的机会，欢迎大家积极报名。


## 介绍

### 目录结构

- AlexNet-torch: 将pytorch代码中与AlexNet图像分类任务有关的代码抽取出来，用于复现
- AlexNet-paddle：基于`AlexNet-torch`进行的AlexNet模型复现。
- notebook：整理PPT过程中遇到过的一些代码小片段，算是一点小随笔了。


注意：这里为了严格保持对齐的步骤，将数据集中的每个类别对应的图片都装在了同一个文件夹中，但是实际上，使用`train_list.txt`这种文本的方式去记录训练数据列表，整体训练文件会更加灵活可控一些，更加具体的实现可以参考：PaddleClas中的数据读取部分。


### 官方模型套件指路。

* 图像分类和识别PaddleClas： https://github.com/PaddlePaddle/PaddleClas
* 文字检测和识别PaddleOCR： https://github.com/PaddlePaddle/PaddleOCR
* 目标检测PaddleDetection： https://github.com/PaddlePaddle/PaddleDetection
* 图像分割PaddleSeg： https://github.com/PaddlePaddle/PaddleSeg
* 生成对抗网络PaddleGAN： https://github.com/PaddlePaddle/PaddleGAN
* 视频分类PaddleVideo： https://github.com/PaddlePaddle/PaddleVideo
* 文本理解PaddleNLP： https://github.com/PaddlePaddle/PaddleNLP


## 欢迎贡献

* 如果您有更多有意思的paddle代码小片段，也欢迎提个pr，上传到`notebook`文件夹，可以是一些自己实现的骚操作，也可以是一些组合的一些api，也可以是自己在使用过程中觉得有意思的功能。
* 如果您有更多已经复现的文章或者方法，如果有兴趣提交的话，可以新建一个文件夹，命名为`xxxx-paddle`，其中`xxxx`表示复现的算法或者名称，将自己的代码放在该文件夹下，然后提交PR即可。
* 生命不息，开源不止，感谢您的使用和关注，希望我们可以一起用飞桨划出一个更有意思的时代。
