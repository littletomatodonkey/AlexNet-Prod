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


def build_paddle_data_pipeline():
    sys.path.insert(0, "./AlexNet_paddle/")
    import AlexNet_paddle.presets as presets
    import AlexNet_paddle.paddlevision as paddlevision
    dataset_test = paddlevision.datasets.ImageFolder(
        "/paddle/data/ILSVRC2012_torch/val/",
        presets.ClassificationPresetEval(
            crop_size=224, resize_size=256))
    test_sampler = paddle.io.SequenceSampler(dataset_test)
    test_batch_sampler = paddle.io.BatchSampler(
        sampler=test_sampler, batch_size=32)
    data_loader_test = paddle.io.DataLoader(
        dataset_test, batch_sampler=test_batch_sampler, num_workers=0)
    sys.path.pop(0)
    return dataset_test, data_loader_test


def build_torch_data_pipeline():
    sys.path.insert(0, "./AlexNet_torch")
    import AlexNet_torch.presets as presets
    import AlexNet_torch.torchvision as torchvision
    dataset_test = torchvision.datasets.ImageFolder(
        "/paddle/data/ILSVRC2012_torch/val/",
        presets.ClassificationPresetEval(
            crop_size=224, resize_size=256))
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=32,
        sampler=test_sampler,
        num_workers=0,
        pin_memory=True)
    sys.path.pop(0)
    return dataset_test, data_loader_test


def test_transform():
    paddle_transform = build_paddle_transform()
    torch_transform = build_torch_transform()
    img = Image.open("./demo_image/ILSVRC2012_val_00006697.JPEG")

    paddle_img = paddle_transform(img)
    torch_img = torch_transform(img)

    np.testing.assert_allclose(paddle_img, torch_img)


def test_data_pipeline():
    diff_helper = ReprodDiffHelper()
    paddle_dataset, paddle_dataloader = build_paddle_data_pipeline()
    torch_dataset, torch_dataloader = build_torch_data_pipeline()

    logger_paddle_data = ReprodLogger()
    logger_torch_data = ReprodLogger()

    logger_paddle_data.add("length", np.array(len(paddle_dataset)))
    logger_torch_data.add("length", np.array(len(torch_dataset)))

    # random choose 5 images and check
    for idx in range(5):
        rnd_idx = np.random.randint(0, len(paddle_dataset))
        logger_paddle_data.add(f"dataset_{idx}",
                               paddle_dataset[rnd_idx][0].numpy())
        logger_torch_data.add(f"dataset_{idx}",
                              torch_dataset[rnd_idx][0].detach().cpu().numpy())

    for idx, (paddle_batch, torch_batch
              ) in enumerate(zip(paddle_dataloader, torch_dataloader)):
        if idx >= 5:
            break
        logger_paddle_data.add(f"dataloader_{idx}", paddle_batch[0].numpy())
        logger_torch_data.add(f"dataloader_{idx}",
                              torch_batch[0].detach().cpu().numpy())

    diff_helper.compare_info(logger_paddle_data.data, logger_torch_data.data)
    diff_helper.report()


if __name__ == "__main__":
    test_data_pipeline()
