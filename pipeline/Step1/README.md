# 使用方法


本部分内容以前向对齐为例，介绍基于`repord_log`工具对齐的检查流程。其中与`reprod_log`工具有关的部分都是需要开发者需要添加的部分。


```shell
# 进入文件夹
cd pipeline/Step1/
# 下载预训练模型
wget -P ../weights https://paddle-model-ecology.bj.bcebos.com/model/alexnet_reprod/alexnet_paddle.pdparams
wget -P ../weights https://paddle-model-ecology.bj.bcebos.com/model/alexnet_reprod/alexnet-owt-7be5be79.pth

# 生成paddle的前向数据
cd AlexNet_paddle/ && python3.7 forward_alexnet.py
# 生成torch的前向数据
cd ../AlexNet_torch && python3.7 forward_alexnet.py
# 对比生成log
cd ..
python3.7 check_step1.py
```

具体地，以PaddlePaddle为例，`forward_alexnet.py`的具体代码如下所示。

```python
import numpy as np
import paddle
# 导入模型
from paddlevision.models.alexnet import alexnet
# 导入reprod_log中的ReprodLogger类
from reprod_log import ReprodLogger

reprod_logger = ReprodLogger()
# 组网并初始化
model = alexnet(pretrained="../../weights/alexnet_paddle.pdparams" num_classes=1000)
model.eval()
# 读入fake data并转换为tensor，这里也可以固定seed在线生成fake data
fake_data = np.load("../../fake_data/fake_data.npy")
fake_data = paddle.to_tensor(fake_data)
# 模型前向
out = model(fake_data)
# 保存前向结果，对于不同的任务，需要开发者添加。
reprod_logger.add("logits", fake_data.cpu().detach().numpy())
reprod_logger.save("forward_paddle.npy")
```

diff检查的代码可以参考：[check_step1.py](./check_step1.py)，具体代码如下所示。

```python
# https://github.com/littletomatodonkey/AlexNet-Prod/blob/master/pipeline/Step1/check_step1.py
# 使用reprod_log排查diff
from reprod_log import ReprodDiffHelper
if __name__ == "__main__":
    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info("AlexNet_torch/forward_torch.npy")
    paddle_info = diff_helper.load_info("AlexNet_paddle/forward_paddle.npy")
    diff_helper.compare_info(torch_info, paddle_info)
    diff_helper.report(path="forward_diff.log")
```

产出日志如下，同时会将check的结果保存在`forward_diff.log`文件中。

```
2021-09-27 10:35:46,172 - reprod_log.utils - INFO - logits:
2021-09-27 10:35:46,173 - reprod_log.utils - INFO -     mean diff: check passed: True, value: 0.0
2021-09-27 10:35:46,173 - reprod_log.utils - INFO - diff check passed
```

平均绝对误差为0，测试通过。
