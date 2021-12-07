## 1 准备部署环境

首先准备docker环境，AIStudio环境已经安装了合适的docker。如果是非AIStudio环境，请[参考文档](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.3/doc/doc_ch/environment.md#2)中的 "1.3.2 Docker环境配置" 安装docker环境。

然后安装Paddle Serving三个安装包，paddle-serving-server，paddle-serving-client 和 paddle-serving-app。

```
wget https://paddle-serving.bj.bcebos.com/test-dev/whl/paddle_serving_server_gpu-0.7.0.post102-py3-none-any.whl
pip3 install paddle_serving_server_gpu-0.7.0.post102-py3-none-any.whl

wget https://paddle-serving.bj.bcebos.com/test-dev/whl/paddle_serving_client-0.7.0-cp37-none-any.whl
pip3 install paddle_serving_client-0.7.0-cp37-none-any.whl

wget https://paddle-serving.bj.bcebos.com/test-dev/whl/paddle_serving_app-0.7.0-py3-none-any.whl
pip3 install paddle_serving_app-0.7.0-py3-none-any.whl
```

Paddle Serving Server更多不同运行环境的whl包下载地址，请参考：[下载页面](https://github.com/PaddlePaddle/Serving/blob/v0.7.0/doc/Latest_Packages_CN.md)

## 2 准备服务化部署模型

下载Alexnet模型，并使用下面命令，将静态图模型转换为服务化部署模型，即serving_server文件夹中的模型。

```python
python3 -m paddle_serving_client.convert --dirname ./alexnet_infer/ --model_filename inference.pdmodel --params_filename inference.pdiparams --serving_server serving_server --serving_client serving_client
```

## 3 启动模型预测服务

以下命令启动模型预测服务：

```bash
python3 web_service.py &
```

服务启动成功的界面如下：

![](../../images/py_serving_startup_visualization.jpg)

## 4 客户端访问服务

以下命令通过客户端访问服务：

```
python3 alex_pipeline_http_client.py
```
访问成功的界面如下：

![](../../images/py_serving_client_results.jpg)


【注意事项】
如果访问不成功，可能设置了代理影响的，可以用下面命令取消代理设置。

```
unset http_proxy
unset https_proxy
```


