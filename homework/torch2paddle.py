import torch
import paddle

from pd.resnet import resnet50


def torch2paddle():
    model = resnet50()
    model_state_dict = model.state_dict()

    torch_path = "./tch/resnet50.pth"
    paddle_path = "./pd/resnet50.pdparams"
    torch_state_dict = torch.load(torch_path)
    fc_names = ["fc"]
    paddle_state_dict = {}
    for k in torch_state_dict:
        v = torch_state_dict[k].detach().cpu().numpy()
        flag = [i in k for i in fc_names]
        if any(flag):
            v = v.transpose()
        k = k.replace("running_var", "_variance")
        k = k.replace("running_mean", "_mean")
        if k not in model_state_dict:
            print(k)
        else:
            paddle_state_dict[k] = v
    paddle.save(paddle_state_dict, paddle_path)
    #diff = list(set(model_state_dict).difference(set(torch_state_dict)))
    #print(diff)
    #diff = list(set(torch_state_dict).difference(set(model_state_dict)))
    #print(diff)

    model.set_dict(paddle_state_dict)


if __name__ == "__main__":
    torch2paddle()
