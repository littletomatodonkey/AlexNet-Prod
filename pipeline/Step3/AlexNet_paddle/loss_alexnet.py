import numpy as np
import paddle
import paddle.nn as nn
from paddlevision.models.alexnet import alexnet

from reprod_log import ReprodLogger

if __name__ == "__main__":
    paddle.set_device("cpu")
    # load model
    # the model is save into ~/.cache/torch/hub/checkpoints/alexnet-owt-4df8aa71.pth

    # def logger
    reprod_logger = ReprodLogger()

    model = alexnet(
        pretrained="../../weights/alexnet_paddle.pdparams", num_classes=1000)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    # read or gen fake data
    fake_data = np.load("../../fake_data/fake_data.npy")
    fake_data = paddle.to_tensor(fake_data)

    fake_label = np.load("../../fake_data/fake_label.npy")
    fake_label = paddle.to_tensor(fake_label)

    # forward
    out = model(fake_data)

    loss = criterion(out, fake_label)
    # 
    reprod_logger.add("loss", loss.cpu().detach().numpy())
    reprod_logger.save("loss_paddle.npy")
