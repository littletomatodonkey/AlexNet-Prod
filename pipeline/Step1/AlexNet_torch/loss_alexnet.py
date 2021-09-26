import numpy as np
import torch
import torch.nn as nn
from torchvision.models.alexnet import alexnet

from reprod_log import ReprodLogger

if __name__ == "__main__":
    # load model
    # the model is save into ~/.cache/torch/hub/checkpoints/alexnet-owt-4df8aa71.pth

    # def logger
    reprod_logger = ReprodLogger()

    criterion = nn.CrossEntropyLoss()

    model = alexnet(pretrained=True, num_classes=1000)
    model.eval()
    # read or gen fake data
    fake_data = np.load("../../fake_data/fake_data.npy")
    fake_data = torch.from_numpy(fake_data)

    fake_label = np.load("../../fake_data/fake_label.npy")
    fake_label = torch.from_numpy(fake_label)

    # forward
    out = model(fake_data)

    loss = criterion(out, fake_label)
    # 
    reprod_logger.add("loss", loss.cpu().detach().numpy())
    reprod_logger.save("loss_torch.npy")
