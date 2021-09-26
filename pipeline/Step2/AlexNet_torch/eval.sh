export PYTHONPATH=.

python3.7 train.py \
    --data-path /paddle/data/ILSVRC2012_torch \
    --test-only \
    --pretrained
