import datetime
import os
import time

import paddle
from paddle import nn
import paddlevision

import presets
import utils

import numpy as np
import random

try:
    from apex import amp
except ImportError:
    amp = None

import numpy as np
from reprod_log import ReprodLogger


def train_one_epoch(model,
                    criterion,
                    optimizer,
                    data_loader,
                    device,
                    epoch,
                    print_freq,
                    apex=False):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        'lr', utils.SmoothedValue(
            window_size=1, fmt='{value}'))
    metric_logger.add_meter(
        'img/s', utils.SmoothedValue(
            window_size=10, fmt='{value}'))

    header = 'Epoch: [{}]'.format(epoch)
    for image, target in metric_logger.log_every(data_loader, print_freq,
                                                 header):
        start_time = time.time()
        output = model(image)
        loss = criterion(output, target)

        if apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.get_lr())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.meters['img/s'].update(batch_size /
                                             (time.time() - start_time))


def train_some_iters(model,
                     criterion,
                     optimizer,
                     fake_data,
                     fake_label,
                     device,
                     epoch,
                     print_freq,
                     apex=False,
                     max_iter=2):
    # needed to avoid network randomness
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        'lr', utils.SmoothedValue(
            window_size=1, fmt='{value}'))
    metric_logger.add_meter(
        'img/s', utils.SmoothedValue(
            window_size=10, fmt='{value}'))

    loss_list = []
    for idx in range(max_iter):
        image = paddle.to_tensor(fake_data)
        target = paddle.to_tensor(fake_label)

        output = model(image)
        loss = criterion(output, target)
        loss.backward()
        # for name, tensor in model.named_parameters():
        #     grad = tensor.grad
        #     print(name, tensor.grad.shape)
        #     break
        optimizer.step()
        optimizer.clear_grad()
        loss_list.append(loss)

    return loss_list


def evaluate(model, criterion, data_loader, device, print_freq=100):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with paddle.no_grad():
        for image, target in metric_logger.log_every(data_loader, print_freq,
                                                     header):
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print(' * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'.format(
        top1=metric_logger.acc1, top5=metric_logger.acc5))
    return metric_logger.acc1.global_avg


def load_data(traindir, valdir, args):
    # Data loading code
    print("Loading data")
    resize_size, crop_size = (342, 299) if args.model == 'inception_v3' else (
        256, 224)

    print("Loading training data")
    st = time.time()
    auto_augment_policy = getattr(args, "auto_augment", None)
    random_erase_prob = getattr(args, "random_erase", 0.0)
    dataset = paddlevision.datasets.ImageFolder(
        traindir,
        presets.ClassificationPresetTrain(
            crop_size=crop_size,
            auto_augment_policy=auto_augment_policy,
            random_erase_prob=random_erase_prob))

    print("Took", time.time() - st)

    print("Loading validation data")
    dataset_test = paddlevision.datasets.ImageFolder(
        valdir,
        presets.ClassificationPresetEval(
            crop_size=crop_size, resize_size=resize_size))

    print("Creating data loaders")
    train_sampler = paddle.io.RandomSampler(dataset)
    test_sampler = paddle.io.SequenceSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    print(args)

    device = paddle.set_device(args.device)

    train_dir = os.path.join(args.data_path, 'train')
    val_dir = os.path.join(args.data_path, 'val')
    dataset, dataset_test, train_sampler, test_sampler = load_data(
        train_dir, val_dir, args)
    train_batch_sampler = paddle.io.BatchSampler(
        sampler=train_sampler, batch_size=args.batch_size)
    data_loader = paddle.io.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=args.workers)
    test_batch_sampler = paddle.io.BatchSampler(
        sampler=test_sampler, batch_size=args.batch_size)
    data_loader_test = paddle.io.DataLoader(
        dataset_test,
        batch_sampler=test_batch_sampler,
        num_workers=args.workers)

    print("Creating model")
    model = paddlevision.models.__dict__[args.model](
        pretrained=args.pretrained)

    criterion = nn.CrossEntropyLoss()

    lr_scheduler = paddle.optimizer.lr.StepDecay(
        args.lr, step_size=args.lr_step_size, gamma=args.lr_gamma)

    opt_name = args.opt.lower()
    if opt_name == 'sgd':
        optimizer = paddle.optimizer.Momentum(
            learning_rate=lr_scheduler,
            momentum=args.momentum,
            parameters=model.parameters(),
            weight_decay=args.weight_decay)
    elif opt_name == 'rmsprop':
        optimizer = paddle.optimizer.RMSprop(
            learning_rate=lr_scheduler,
            momentum=args.momentum,
            parameters=model.parameters(),
            weight_decay=args.weight_decay,
            eps=0.0316,
            alpha=0.9)
    else:
        raise RuntimeError(
            "Invalid optimizer {}. Only SGD and RMSprop are supported.".format(
                args.opt))

    model_without_ddp = model

    if args.resume:
        layer_state_dict = paddle.load(os.path.join(args.resume, '.pdparams'))
        model_without_ddp.set_state_dict(layer_state_dict)
        opt_state_dict = paddle.load(os.path.join(args.resume, '.pdopt'))
        optimizer.load_state_dict(opt_state_dict)

    if args.test_only:
        top1 = evaluate(model, criterion, data_loader_test, device=device)
        return top1

    print("Start training")
    fake_data = np.load("../../fake_data/fake_data.npy")
    fake_label = np.load("../../fake_data/fake_label.npy")

    loss_list = train_some_iters(
        model,
        criterion,
        optimizer,
        fake_data,
        fake_label,
        device,
        0,
        args.print_freq,
        max_iter=5)

    print(loss_list)
    return loss_list


def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(
        description='PaddlePaddle Classification Training', add_help=add_help)

    parser.add_argument('--data-path', default='./data', help='dataset')
    parser.add_argument('--model', default='alexnet', help='model')
    parser.add_argument('--device', default='cpu', help='device')
    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument(
        '--epochs',
        default=90,
        type=int,
        metavar='N',
        help='number of total epochs to run')
    parser.add_argument(
        '-j',
        '--workers',
        default=16,
        type=int,
        metavar='N',
        help='number of data loading workers (default: 16)')
    parser.add_argument('--opt', default='sgd', type=str, help='optimizer')
    parser.add_argument(
        '--lr', default=0.00125, type=float, help='initial learning rate')
    parser.add_argument(
        '--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument(
        '--wd',
        '--weight-decay',
        default=1e-4,
        type=float,
        metavar='W',
        help='weight decay (default: 1e-4)',
        dest='weight_decay')
    parser.add_argument(
        '--lr-step-size',
        default=30,
        type=int,
        help='decrease lr every step-size epochs')
    parser.add_argument(
        '--lr-gamma',
        default=0.1,
        type=float,
        help='decrease lr by a factor of lr-gamma')
    parser.add_argument(
        '--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument(
        '--start-epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true", )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true", )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo")
    parser.add_argument(
        '--auto-augment',
        default=None,
        help='auto augment policy (default: None)')
    parser.add_argument(
        '--random-erase',
        default=0.0,
        type=float,
        help='random erasing probability (default: 0.0)')

    # Mixed precision training parameters
    parser.add_argument(
        '--apex',
        action='store_true',
        help='Use apex for mixed precision training')
    parser.add_argument(
        '--apex-opt-level',
        default='O1',
        type=str,
        help='For apex mixed precision training'
        'O0 for FP32 training, O1 for mixed precision training.'
        'For further detail, see https://github.com/NVIDIA/apex/tree/master/examples/imagenet'
    )

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    loss_list = main(args)
    reprod_logger = ReprodLogger()
    for idx, loss in enumerate(loss_list):
        reprod_logger.add(f"loss_{idx}", loss.detach().cpu().numpy())
    reprod_logger.save("bp_align_paddle.npy")
