# 使用方法

## 数据集和数据加载对齐步骤

* 使用下面的命令，判断数据预处理以及数据集是否构建正确。

```shell
python test_data.py
```

显示出以下内容，Dataset以及Dataloader的长度和内容diff均满足小于指定阈值，可以认为复现成功。

```
2021/10/12 23:21:30] root INFO: length:
[2021/10/12 23:21:30] root INFO:        mean diff: check passed: True, value: 0.0
[2021/10/12 23:21:30] root INFO: dataset_0:
[2021/10/12 23:21:30] root INFO:        mean diff: check passed: True, value: 0.0
[2021/10/12 23:21:30] root INFO: dataset_1:
[2021/10/12 23:21:30] root INFO:        mean diff: check passed: True, value: 0.0
[2021/10/12 23:21:30] root INFO: dataset_2:
[2021/10/12 23:21:30] root INFO:        mean diff: check passed: True, value: 0.0
[2021/10/12 23:21:30] root INFO: dataset_3:
[2021/10/12 23:21:30] root INFO:        mean diff: check passed: True, value: 0.0
[2021/10/12 23:21:30] root INFO: dataset_4:
[2021/10/12 23:21:30] root INFO:        mean diff: check passed: True, value: 0.0
[2021/10/12 23:21:30] root INFO: dataloader_0:
[2021/10/12 23:21:30] root INFO:        mean diff: check passed: True, value: 0.0
[2021/10/12 23:21:30] root INFO: dataloader_1:
[2021/10/12 23:21:30] root INFO:        mean diff: check passed: True, value: 0.0
[2021/10/12 23:21:30] root INFO: dataloader_2:
[2021/10/12 23:21:30] root INFO:        mean diff: check passed: True, value: 0.0
[2021/10/12 23:21:30] root INFO: dataloader_3:
[2021/10/12 23:21:30] root INFO:        mean diff: check passed: True, value: 0.0
[2021/10/12 23:21:30] root INFO: dataloader_4:
[2021/10/12 23:21:30] root INFO:        mean diff: check passed: True, value: 0.0
[2021/10/12 23:21:30] root INFO: diff check passed
```


## 数据评估对齐流程

运行下面的命令，验证数据集评估是否正常。

```shell
# 生成paddle的前向数据
cd AlexNet_paddle/
sh eval.sh

# 生成torch的前向数据
cd ../AlexNet_torch
sh eval.sh

# 对比生成log
cd ..
python3.7 check_step2.py
```

最终结果输出如下，top1精度diff为0，小于阈值，结果前向验证，
```
[2021/10/13 00:05:55] root INFO: top1:
[2021/10/13 00:05:55] root INFO:        mean diff: check passed: True, value: 0.0
[2021/10/13 00:05:55] root INFO: diff check passed
```
