import os
import paddle
from paddle import inference
import numpy as np
from PIL import Image

from reprod_log import ReprodLogger
from preprocess import ResizeImage, CenterCropImage, NormalizeImage, ToCHW, Compose


def load_predictor(model_file_path, params_file_path, args):
    config = inference.Config(model_file_path, params_file_path)
    if args.use_gpu:
        config.enable_use_gpu(1000, 0)
        if args.use_tensorrt:
            if hasattr(args, "precision"):
                if args.precision == "fp16" and args.use_tensorrt:
                    precision = inference.PrecisionType.Half
                elif args.precision == "int8":
                    precision = inference.PrecisionType.Int8
                else:
                    precision = inference.PrecisionType.Float32
            else:
                precision = inference.PrecisionType.Float32

            config.enable_tensorrt_engine(
                workspace_size=1 << 30,
                precision_mode=precision,
                max_batch_size=args.max_batch_size,
                min_subgraph_size=args.min_subgraph_size)
    else:
        config.disable_gpu()
        if args.use_mkldnn:
            config.enable_mkldnn()
            config.set_cpu_math_library_num_threads(args.cpu_threads)

    # enable memory optim
    config.enable_memory_optim()
    config.disable_glog_info()

    config.switch_use_feed_fetch_ops(False)
    config.switch_ir_optim(True)

    # create predictor
    predictor = inference.create_predictor(config)
    return predictor, config


def get_args(add_help=True):
    import argparse

    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser(
        description="PaddlePaddle Classification Training", add_help=add_help)

    parser.add_argument(
        "--model-dir", default=None, help="inference model dir")
    parser.add_argument(
        "--use-gpu", default=False, type=str2bool, help="use_gpu")
    parser.add_argument(
        "--use-tensorrt", default=False, type=str2bool, help="use_trt")
    parser.add_argument(
        "--use-mkldnn", default=False, type=str2bool, help="use_mkldnn")
    parser.add_argument("--precision", default="fp32", help="precision")
    parser.add_argument(
        "--min-subgraph-size", default=15, type=int, help="min_subgraph_size")
    parser.add_argument(
        "--max-batch-size", default=16, type=int, help="max_batch_size")
    parser.add_argument(
        "--cpu-threads", default=10, type=int, help="cpu-threads")
    parser.add_argument("--batch-size", default=1, type=int, help="batch size")

    parser.add_argument(
        "--resize-size", default=256, type=int, help="resize_size")
    parser.add_argument("--crop-size", default=224, type=int, help="crop_szie")
    parser.add_argument("--img-path", default="./images/demo.jpg")

    parser.add_argument(
        "--benchmark", default=True, type=str2bool, help="benchmark")
    parser.add_argument("--warmup", default=0, type=int, help="warmup iter")

    args = parser.parse_args()
    return args


def get_infer_gpuid():
    cmd = "env | grep CUDA_VISIBLE_DEVICES"
    env_cuda = os.popen(cmd).readlines()
    if len(env_cuda) == 0:
        return 0
    else:
        gpu_id = env_cuda[0].strip().split("=")[1]
        return int(gpu_id[0])


def predict(args):
    predictor, config = load_predictor(
        os.path.join(args.model_dir, "inference.pdmodel"),
        os.path.join(args.model_dir, "inference.pdiparams"), args)

    input_names = predictor.get_input_names()
    input_tensor = predictor.get_input_handle(input_names[0])

    output_names = predictor.get_output_names()
    output_tensor = predictor.get_output_handle(output_names[0])

    assert args.batch_size == 1, "batch size just supports 1 now."

    # init benchmark
    if args.benchmark:
        import auto_log
        pid = os.getpid()
        gpu_id = get_infer_gpuid()
        autolog = auto_log.AutoLogger(
            model_name="classification",
            model_precision=args.precision,
            batch_size=args.batch_size,
            data_shape="dynamic",
            save_path=None,
            inference_config=config,
            pids=pid,
            process_name=None,
            gpu_ids=gpu_id if args.use_gpu else None,
            time_keys=[
                "preprocess_time", "inference_time", "postprocess_time"
            ],
            warmup=0,
            logger=None)

    eval_transforms = Compose([
        ResizeImage(args.resize_size), CenterCropImage(args.crop_size),
        NormalizeImage(), ToCHW()
    ])

    # wamrup
    if args.warmup > 0:
        for _ in range(args.warmup):
            x = paddle.rand([1, 3, args.crop_size, args.crop_size])
            input_tensor.copy_from_cpu(x)
            predictor.run()
            output = output_tensor.copy_to_cpu()

    if args.benchmark:
        autolog.times.start()

    # inference
    with open(args.img_path, "rb") as f:
        img = Image.open(f)
        img = img.convert("RGB")

    img = eval_transforms(img)
    img = np.expand_dims(img, axis=0)
    input_tensor.copy_from_cpu(img)

    if args.benchmark:
        autolog.times.stamp()

    predictor.run()
    output = output_tensor.copy_to_cpu()

    if args.benchmark:
        autolog.times.stamp()

    output = output.flatten()
    class_id = output.argmax()

    if args.benchmark:
        autolog.times.stamp()

    prob = output[class_id]
    print(f"image_name: {args.img_path}, class_id: {class_id}, prob: {prob}")

    if args.benchmark:
        autolog.times.end(stamp=True)
        autolog.report()
    return output


if __name__ == "__main__":
    args = get_args()
    output = predict(args)

    reprod_logger = ReprodLogger()
    reprod_logger.add("output", output)
    reprod_logger.save("output_inference_engine.npy")
