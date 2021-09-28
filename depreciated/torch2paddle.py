import numpy as np
import torch
from torchvision import models
from torchsummary import summary
import paddle

from alexnet_torch import alexnet_torch
from alexnet_paddle import alexnet_paddle


def model_summary():
    model = alexnet_torch().cuda()
    checkpoint = torch.load('model_89.pth')
    model.load_state_dict(checkpoint['model'])
    summary(model, (3, 224, 224))
    for name in model.state_dict():
        print(name)


def show_layer_name():
    model = alexnet_torch().cuda()
    for name in model.state_dict():
        print(name)


def transfer():
    input_fp = "model.pth"
    output_fp = "model.pdparams"
    torch_dict = torch.load(input_fp)['model']
    paddle_dict = {}
    fc_names = [
        "classifier.1.weight", "classifier.4.weight", "classifier.6.weight"
    ]
    for key in torch_dict:
        weight = torch_dict[key].cpu().numpy()
        flag = [i in key for i in fc_names]
        if any(flag):
            print("weight {} need to be trans".format(key))
            weight = weight.transpose()
        paddle_dict[key] = weight
    paddle.save(paddle_dict, output_fp)


def test_forward():
    model_torch = alexnet_torch()
    model_paddle = alexnet_paddle()
    model_torch.eval()
    model_paddle.eval()
    torch_checkpoint = torch.load('model.pth')
    model_torch.load_state_dict(torch_checkpoint['model'])
    paddle_checkpoint = paddle.load('model.pdparams')
    model_paddle.set_state_dict(paddle_checkpoint)

    x = np.random.randn(1, 3, 224, 224)
    input_torch = torch.tensor(x, dtype=torch.float32)
    out_torch = model_torch(input_torch)

    input_paddle = paddle.to_tensor(x, dtype='float32')
    out_paddle = model_paddle(input_paddle)

    print('torch result:{}'.format(out_torch))
    print('paddle result:{}'.format(out_paddle))


if __name__ == '__main__':
    #model_summary()
    #transfer()
    test_forward()
    #show_layer_name()
