import numpy as np
import paddle
import torch

from pd.resnet import resnet50 as p_r50
from tch.resnet import resnet50 as t_r50


def test_forward():
    paddle_model = p_r50()
    paddle_model.eval()
    paddle_state_dict = paddle.load("./pd/resnet50.pdparams")
    paddle_model.set_dict(paddle_state_dict)

    torch_model = t_r50()
    torch_model.eval()
    torch_state_dict = torch.load("./tch/resnet50.pth")
    torch_model.load_state_dict(torch_state_dict)

    inputs = np.load("demo_data.npy")
    paddle_out = paddle_model(paddle.to_tensor(
        inputs, dtype="float32")).numpy()
    torch_out = torch_model(torch.tensor(
        inputs, dtype=torch.float32)).cpu().detach().numpy()
    np.testing.assert_allclose(paddle_out, torch_out, atol=1e-5, rtol=1e-2)


if __name__ == "__main__":
    test_forward()
