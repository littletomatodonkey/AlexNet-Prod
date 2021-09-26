# 使用方法

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
