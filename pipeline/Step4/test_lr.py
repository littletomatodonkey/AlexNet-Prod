import os
import sys
import cv2
from PIL import Image
import numpy as np
import paddle
import torch
from reprod_log import ReprodLogger, ReprodDiffHelper


def build_paddle_opt():
    sys.path.insert(0, "./AlexNet_paddle/")
    from paddlevision.models.alexnet import alexnet
    model = alexnet(pretrained=False, num_classes=1000)
    lr_scheduler = paddle.optimizer.lr.StepDecay(0.1, step_size=30, gamma=0.1)
    optimizer = paddle.optimizer.Momentum(
        learning_rate=lr_scheduler,
        momentum=0.9,
        parameters=model.parameters(),
        weight_decay=1e-4)
    sys.path.pop(0)
    return lr_scheduler, optimizer


def build_torch_opt():
    sys.path.insert(0, "./AlexNet_torch/")
    from torchvision.models.alexnet import alexnet
    model = alexnet(pretrained=False, num_classes=1000)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=0.1,
                                momentum=0.9,
                                weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=30, gamma=0.1)

    sys.path.pop(0)

    return lr_scheduler, optimizer


def test_lr_opt():
    diff_helper = ReprodDiffHelper()
    paddle_lr, paddle_opt = build_paddle_opt()
    torch_lr, torch_opt = build_torch_opt()

    logger_paddle_data = ReprodLogger()
    logger_torch_data = ReprodLogger()

    paddle_lr_list = []
    torch_lr_list = []
    
    for idx in range(90):
        paddle_lr_list.append(paddle_lr.get_lr())
        torch_lr_list.append(torch_lr.get_lr())

    logger_paddle_data.add("lr", np.array(paddle_lr_list))
    logger_torch_data.add("lr", np.array(torch_lr_list))

    diff_helper.compare_info(logger_paddle_data.data, logger_torch_data.data)
    diff_helper.report()


if __name__ == "__main__":
    paddle.set_device("cpu")
    test_lr_opt()
