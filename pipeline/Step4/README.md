# 使用方法

### 学习率对齐验证

运行下面的命令，检查学习率模块设置是否正确。

```shell
python test_lr.py
```


### 反向对齐操作方法

#### 代码讲解

以PaddlePaddle为例，训练流程核心代码如下所示。每个iter中输入相同的fake data与fake label，计算loss，进行梯度反传与参数更新，将loss批量返回，用于后续的验证。

```python
# https://github.com/littletomatodonkey/AlexNet-Prod/blob/1f422872ed9831ea25d1dc0c989fa5452ae67de8/pipeline/Step4/AlexNet_paddle/train.py#L65
def train_some_iters(model,
                     criterion,
                     optimizer,
                     fake_data,
                     fake_label,
                     device,
                     epoch,
                     print_freq,
                     apex=False,
                     max_iter=2):
    # needed to avoid network randomness
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        'lr', utils.SmoothedValue(
            window_size=1, fmt='{value}'))
    metric_logger.add_meter(
        'img/s', utils.SmoothedValue(
            window_size=10, fmt='{value}'))

    loss_list = []
    for idx in range(max_iter):
        image = paddle.to_tensor(fake_data)
        target = paddle.to_tensor(fake_label)

        output = model(image)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        loss_list.append(loss)

    return loss_list
```


#### 操作方法

运行下面的命令，基于fake data与fake label，依次生成若干轮loss数据并保存，使用`reprod_log`工具进行diff排查。

```shell
# 生成paddle的前向数据
cd AlexNet_paddle/
sh train.sh

# 生成torch的前向数据
cd ../AlexNet_torch
sh train.sh

# 对比生成log
cd ..
python3.7 check_step4.py
```

最终输出结果如下，同时会保存在文件`bp_align_diff.log`中。

```
[2021/10/13 13:03:05] root INFO: loss_0: 
[2021/10/13 13:03:05] root INFO:        mean diff: check passed: True, value: 9.5367431640625e-07
[2021/10/13 13:03:05] root INFO: loss_1: 
[2021/10/13 13:03:05] root INFO:        mean diff: check passed: True, value: 4.76837158203125e-07
[2021/10/13 13:03:05] root INFO: loss_2: 
[2021/10/13 13:03:05] root INFO:        mean diff: check passed: True, value: 0.0
[2021/10/13 13:03:05] root INFO: loss_3: 
[2021/10/13 13:03:05] root INFO:        mean diff: check passed: True, value: 1.1548399925231934e-07
[2021/10/13 13:03:05] root INFO: loss_4: 
[2021/10/13 13:03:05] root INFO:        mean diff: check passed: True, value: 3.7834979593753815e-10
[2021/10/13 13:03:05] root INFO: diff check passed
```

前面5轮的loss diff均小于阈值(1e-6)，check通过。
