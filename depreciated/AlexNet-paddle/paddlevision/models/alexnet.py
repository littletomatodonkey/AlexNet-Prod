import os
import paddle
import paddle.nn as nn
from typing import Any

__all__ = ['AlexNet', 'alexnet']


class AlexNet(nn.Layer):
    def __init__(self, num_classes: int=1000) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2D(
                3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2D(
                kernel_size=3, stride=2),
            nn.Conv2D(
                64, 192, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2D(
                kernel_size=3, stride=2),
            nn.Conv2D(
                192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2D(
                384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2D(
                256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2D(
                kernel_size=3, stride=2), )
        self.avgpool = nn.AdaptiveAvgPool2D((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes), )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = paddle.flatten(x, 1)
        x = self.classifier(x)
        return x


def load_dygraph_pretrain(model, path=None):
    if not (os.path.isdir(path) or os.path.exists(path + '.pdparams')):
        raise ValueError("Model pretrain path {} does not "
                         "exists.".format(path))
    param_state_dict = paddle.load(path + ".pdparams")
    model.set_dict(param_state_dict)
    return


def alexnet(pretrained: bool=False, **kwargs: Any) -> AlexNet:
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    The required minimum input size of the model is 63x63.

    Args:
        pretrained (str): Pre-trained parameters of the model on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        load_dygraph_pretrain(model, pretrained)
    return model
