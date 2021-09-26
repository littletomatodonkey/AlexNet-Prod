import numpy as np
import paddle

from paddlevision.models.alexnet import alexnet

from reprod_log import ReprodLogger

if __name__ == "__main__":
    # load model
    # the model is save into ~/.cache/torch/hub/checkpoints/alexnet-owt-4df8aa71.pth

    # def logger
    reprod_logger = ReprodLogger()

    model = alexnet(
        pretrained="../../weights/alexnet_paddle.pdparams", num_classes=1000)
    model.eval()

    # read or gen fake data
    fake_data = np.load("../../fake_data/fake_data.npy")
    fake_data = paddle.to_tensor(fake_data)
    # forward
    out = model(fake_data)
    # 
    reprod_logger.add("logits", fake_data.cpu().detach().numpy())
    reprod_logger.save("forward_paddle.npy")
