#!/bin/bash

gpu=0
arch=deeplab_large_fov
name=YOUR_TRAIN_NAME
dataset="PascalVOC"
data_root="/srv/PascalVOC/VOCdevkit/VOC2012/"
gt_root="datalist/PascalVOC/localization_cues.pickle"
batch=15
gamma=0.1
stepsize=2500
display=50
max_iter=10000
wd=5e-4
snapshot=1000
lr=0.001

resume='vgg16_20M_custom.pth'

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