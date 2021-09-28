import os
import sys
import cv2
from PIL import Image
import numpy as np
import paddle
import torch
from reprod_log import ReprodLogger, ReprodDiffHelper


def build_paddle_transform():
    sys.path.insert(0, "./AlexNet_paddle/")
    import AlexNet_paddle.presets as presets
    paddle_transform = presets.ClassificationPresetEval(
        crop_size=224,
        resize_size=256, )
    sys.path.pop(0)
    return paddle_transform


def build_torch_transform():
    sys.path.insert(0, "./AlexNet_torch/")
    import AlexNet_torch.presets as presets
    torch_transform = presets.ClassificationPresetEval(
        crop_size=224,
        resize_size=256, )
    sys.path.pop(0)

    return torch_transform


def build_paddle_dataset():
    sys.path.insert(0, "./AlexNet_paddle/")
    import AlexNet_paddle.presets as presets
    import AlexNet_paddle.paddlevision as paddlevision
    dataset_test = paddlevision.datasets.ImageFolder(
        "/paddle/data/ILSVRC2012_torch/val/",
        presets.ClassificationPresetEval(
            crop_size=224, resize_size=256))
    sys.path.pop(0)
    return dataset_test


def build_torch_dataset():
    sys.path.insert(0, "./AlexNet_torch")
    import AlexNet_torch.presets as presets
    import AlexNet_torch.torchvision as torchvision
    dataset_test = torchvision.datasets.ImageFolder(
        "/paddle/data/ILSVRC2012_torch/val/",
        presets.ClassificationPresetEval(
            crop_size=224, resize_size=256))
    sys.path.pop(0)
    return dataset_test


def test_transform():
    paddle_transform = build_paddle_transform()
    torch_transform = build_torch_transform()
    img = Image.open("./demo_image/ILSVRC2012_val_00006697.JPEG")

    paddle_img = paddle_transform(img)
    torch_img = torch_transform(img)

    np.testing.assert_allclose(paddle_img, torch_img)


def test_dataset():
    diff_helper = ReprodDiffHelper()
    paddle_dataset = build_paddle_dataset()
    torch_dataset = build_torch_dataset()

    logger_paddle_data = ReprodLogger()
    logger_torch_data = ReprodLogger()

    logger_paddle_data.add("length", np.array(len(paddle_dataset)))
    logger_torch_data.add("length", np.array(len(torch_dataset)))

    # random choose 5 images and check
    for idx in range(5):
        rnd_idx = np.random.randint(0, len(paddle_dataset))
        logger_paddle_data.add(f"data_{idx}",
                               paddle_dataset[rnd_idx][0].numpy())
        logger_torch_data.add(f"data_{idx}",
                              torch_dataset[rnd_idx][0].detach().cpu().numpy())

    diff_helper.compare_info(logger_paddle_data.data, logger_torch_data.data)
    diff_helper.report()


if __name__ == "__main__":
    test_dataset()
