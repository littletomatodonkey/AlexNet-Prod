# 使用方法

## 数据集和数据加载对齐步骤

* 使用下面的命令，判断数据预处理以及数据集是否构建正确。

```shell
tar -xf lite_data.tar
python test_data.py
```

显示出以下内容，Dataset以及Dataloader的长度和内容diff均满足小于指定阈值，可以认为复现成功。

```
[2021/12/04 06:34:43] root INFO: length:
[2021/12/04 06:34:43] root INFO:    mean diff: check passed: True, value: 0.0
[2021/12/04 06:34:43] root INFO: dataset_0:
[2021/12/04 06:34:43] root INFO:    mean diff: check passed: True, value: 0.0
[2021/12/04 06:34:43] root INFO: dataset_1:
[2021/12/04 06:34:43] root INFO:    mean diff: check passed: True, value: 0.0
[2021/12/04 06:34:43] root INFO: dataset_2:
[2021/12/04 06:34:43] root INFO:    mean diff: check passed: True, value: 0.0
[2021/12/04 06:34:43] root INFO: dataset_3:
[2021/12/04 06:34:43] root INFO:    mean diff: check passed: True, value: 0.0
[2021/12/04 06:34:43] root INFO: dataset_4:
[2021/12/04 06:34:43] root INFO:    mean diff: check passed: True, value: 0.0
[2021/12/04 06:34:43] root INFO: dataloader_0:
[2021/12/04 06:34:43] root INFO:    mean diff: check passed: True, value: 0.0
[2021/12/04 06:34:43] root INFO: dataloader_1:
[2021/12/04 06:34:43] root INFO:    mean diff: check passed: True, value: 0.0
[2021/12/04 06:34:43] root INFO: dataloader_2:
[2021/12/04 06:34:43] root INFO:    mean diff: check passed: True, value: 0.0
[2021/12/04 06:34:43] root INFO: dataloader_3:
[2021/12/04 06:34:43] root INFO:    mean diff: check passed: True, value: 0.0
[2021/12/04 06:34:43] root INFO: diff check passed
```


## 数据评估对齐流程

### 评估代码和修改内容说明

Pytorch准确率评估指标代码如下。

```python
# https://github.com/littletomatodonkey/AlexNet-Prod/blob/ea49142949e891e2523d5c44e01539900d5b6e70/pipeline/Step2/AlexNet_torch/utils.py#L162
def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res
```

对应地，PaddlePaddle评估指标代码如下

```python
# https://github.com/littletomatodonkey/AlexNet-Prod/blob/master/pipeline/Step2/AlexNet_paddle/utils.py#L145
def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with paddle.no_grad():
        maxk = max(topk)
        batch_size = target.shape[0]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.equal(target)

        res = []
        for k in topk:
            correct_k = correct.astype(paddle.int32)[:k].flatten().sum(
                dtype='float32')
            res.append(correct_k * (100.0 / batch_size))
        return res
```

具体地，对于AlexNet复现，找到其中的预测评估逻辑，在评估完成之后获取返回值，记录在`metric_paddle.npy`文件中（代码中已经修改好了，这里仅用于说明，无需重复修改）。

```python
...
def main(args):
    if args.test_only:
        top1 = evaluate(model, criterion, data_loader_test, device=device)
        return top1
...
# 打开main test-only选项，仅测试评估流程
if __name__ == "__main__":
    args = get_args_parser().parse_args()
    top1 = main(args)
    reprod_logger = ReprodLogger()
    reprod_logger.add("top1", np.array([top1]))
    reprod_logger.save("metric_paddle.npy")
```

### 操作步骤

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
