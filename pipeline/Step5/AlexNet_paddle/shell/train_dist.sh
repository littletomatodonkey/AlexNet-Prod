export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python3.7 -m paddle.distributed.launch \
    --gpus="0,1,2,3,4,5,6,7" \
    train.py \
    --data-path /paddle/data/ILSVRC2012 \
    --lr 0.01 \
    --batch-size 32 \
    --output-dir "./output/" \
    --epochs 120 \
