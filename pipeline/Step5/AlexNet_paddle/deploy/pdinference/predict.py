import os
import paddle
from paddle import inference
import numpy as np
from PIL import Image

from reprod_log import ReprodLogger

from preprocess import ResizeImage, CenterCropImage, NormalizeImage, ToCHW, Compose


def load_predictor(
        model_file_path,
        params_file_path,
        use_gpu=True,
        use_mkldnn=True, ):
    config = inference.Config(model_file_path, params_file_path)
    if args.use_gpu:
        config.enable_use_gpu(1000, 0)
    else:
        config.disable_gpu()
        if use_mkldnn:
            config.use_mkldnn()
            config.set_cpu_math_library_num_threads(10)

    # enable memory optim
    config.enable_memory_optim()
    config.disable_glog_info()

    config.switch_use_feed_fetch_ops(False)
    config.switch_ir_optim(True)

    # create predictor
    predictor = inference.create_predictor(config)
    return predictor


def get_args(add_help=True):
    import argparse

    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser(
        description='PaddlePaddle Classification Training', add_help=add_help)

    parser.add_argument(
        '--model-dir', default=None, help='inference model dir')
    parser.add_argument(
        '--use_gpu', default=str2bool, type=bool, help='use_gpu')
    parser.add_argument(
        '--use_mkldnn', default=str2bool, type=bool, help='use_mkldnn')
    parser.add_argument('--resize-size', default=256, help='resize_size')
    parser.add_argument('--crop-size', default=224, help='crop_szie')
    parser.add_argument(
        '--img-path', default='./images/demo.jpg', help='path where to save')
    parser.add_argument('--num-classes', default=1000, help='num_classes')
    args = parser.parse_args()
    return args


def predict(args):
    predictor = load_predictor(
        os.path.join(args.model_dir, "inference.pdmodel"),
        os.path.join(args.model_dir, "inference.pdiparams"),
        use_gpu=args.use_gpu,
        use_mkldnn=args.use_mkldnn)

    input_names = predictor.get_input_names()
    input_tensor = predictor.get_input_handle(input_names[0])

    output_names = predictor.get_output_names()
    output_tensor = predictor.get_output_handle(output_names[0])

    eval_transforms = Compose([
        ResizeImage(args.resize_size), CenterCropImage(args.crop_size),
        NormalizeImage(), ToCHW()
    ])

    with open(args.img_path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')

    img = eval_transforms(img)
    img = np.expand_dims(img, axis=0)

    input_tensor.copy_from_cpu(img)
    predictor.run()
    output = output_tensor.copy_to_cpu()
    output = output.flatten()

    class_id = output.argmax()
    prob = output[class_id]
    print(f"class_id: {class_id}, prob: {prob}")
    return output


if __name__ == "__main__":
    args = get_args()
    output = predict(args)

    reprod_logger = ReprodLogger()
    reprod_logger.add("output", output)
    reprod_logger.save("output_inference_engine.npy")
