#!/bin/bash

gpu=0
arch=deeplab_large_fov
name=coco_full1_adl1_02
dataset="COCO"
data_root="/srv/COCO/"
gt_root="datalist/COCO/cues/coco_full1_adl1_02.pickle"
batch=32
gamma=0.1
stepsize=2500
display=50
max_iter=10000
wd=5e-4
snapshot=1000
lr=0.001

resume='train_log/pretrained_model/vgg16_20M_custom.pth'

CUDA_VISIBLE_DEVICES=${gpu} python main.py \
    --arch ${arch} \
    --name ${name} \
    --data ${data_root} \
    --gt-root ${gt_root} \
    --dataset ${dataset} \
    --max-iter ${max_iter} \
    --snapshot ${snapshot} \
    --lr-decay ${stepsize} \
    --batch-size ${batch} \
    --lr ${lr} \
    --wd ${wd} \
    --resume ${resume}