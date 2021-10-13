# 使用方法

首先运行下面的python代码，生成`train_align_benchmark.npy`文件，作为训练对齐的基准文件。

```python
# top1-acc 指标来源：https://pytorch.org/hub/pytorch_vision_alexnet/
import numpy as np
reprod_logger = ReprodLogger()
reprod_logger.add("top1", np.array([0.5655], dtype="float32"))
reprod_logger.save("train_align_benchmark.npy")
```

然后运行下面的代码，运行训练脚本；之后使用`check_step5.py`进行精度diff验证。

```shell
cd AlexNet_paddle/
sh train.sh

# 对比生成log
cd ..
python3.7 check_step5.py
```

这里需要注意的是，由于是精度对齐，ImageNet数据集的精度diff在0.15%以内时，可以认为对齐，因此将`diff_threshold`参数修改为了`0.0015`。

```
[2021/10/13 15:54:22] root INFO: top1:
[2021/10/13 15:54:22] root INFO:        mean diff: check passed: True, value: 0.001100003719329834
[2021/10/13 15:54:22] root INFO: diff check passed
```

最终diff为`0.00110`，小于阈值标准，检查通过。
