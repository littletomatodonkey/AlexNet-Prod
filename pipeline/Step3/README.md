# 使用方法

## 代码解析

以PaddlePaddle为例，下面为定义模型、计算loss并保存的代码。

```python
# loss_alexnet.py
if __name__ == "__main__":
    paddle.set_device("cpu")
    # def logger
    reprod_logger = ReprodLogger()

    model = alexnet(
        pretrained="../../weights/alexnet_paddle.pdparams", num_classes=1000)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    # 读取fake data和fake label
    fake_data = np.load("../../fake_data/fake_data.npy")
    fake_data = paddle.to_tensor(fake_data)

    fake_label = np.load("../../fake_data/fake_label.npy")
    fake_label = paddle.to_tensor(fake_label)

    # forward
    out = model(fake_data)
    # 计算loss的值
    loss = criterion(out, fake_label)
    # 记录loss到文件中
    reprod_logger.add("loss", loss.cpu().detach().numpy())
    reprod_logger.save("loss_paddle.npy")
```

记录loss并保存在`loss_paddle.npy`文件中。


## 操作步骤

* 具体操作步骤如下所示。


```shell
# 生成paddle的前向loss结果
cd AlexNet_paddle/
python3.7 loss_alexnet.py

# 生成torch的前向loss结果
cd ../AlexNet_torch
python3.7 loss_alexnet.py

# 对比生成log
cd ..
python3.7 check_step3.py
```

`check_step3.py`的输出结果如下所示，同时也会保存在`loss_diff.log`文件中。

```
2021-09-27 11:29:40,692 - reprod_log.utils - INFO - loss:
2021-09-27 11:29:40,692 - reprod_log.utils - INFO -     mean diff: check passed: True, value: 9.5367431640625e-07
2021-09-27 11:29:40,692 - reprod_log.utils - INFO - diff check passed
```

diff为0，check通过。
