# 安装说明

* 下载代码库

```shell
git clone https://github.com/littletomatodonkey/AlexNet-Prod.git
```

* 进入文件夹，安装requirements

```shell
pip3.7 install -r requirements.txt
```

* 安装PaddlePaddle与PyTorch

```shell
# CPU版本的PaddlePaddle
pip3.7 install paddlepaddle==0.0.0 -f https://www.paddlepaddle.org.cn/whl/linux/cpu-mkl/develop.html
# 如果希望安装GPU版本的PaddlePaddle，可以使用下面的命令
# python -m pip install paddlepaddle-gpu==0.0.0.post102 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
# 安装PyTorch
pip3.7 install torch
```

**注意**: 本项目依赖于paddlepaddle-dev版本，安装时需要注意。
