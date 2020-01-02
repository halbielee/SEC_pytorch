#!/bin/bash

arch='deeplab_large_fov'
image_list='datalist/PascalVOC/val_id.txt'
image_path='/srv/PascalVOC/VOCdevkit/VOC2012/'
log_path='train_log/caffe_trained'
pred_path=${log_path}/pred_images
trained=${log_path}/model_iter_8000.pth

python test_multiprocess.py \
  --arch ${arch} \
  --trained ${trained} \
  --image-list ${image_list} \
  --image-path ${image_path} \
  --pred-path ${pred_path} \
  --gpu 0 \
  --num-gpu 1 \
  --split-size 8 \
  --smooth \
