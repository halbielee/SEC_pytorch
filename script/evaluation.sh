#!/bin/bash

image_list='datalist/PascalVOC/val_id.txt'
gt_path='/srv/PascalVOC/VOCdevkit/VOC2012/SegmentationClassAug/'
log_path='train_log/caffe_trained'
pred_path=${log_path}/pred_images
save_name=${log_path}/evaluation_result.txt

python evaluation.py \
  --image-list ${image_list} \
  --pred-path ${pred_path} \
  --gt-path ${gt_path} \
  --save-name ${save_name} \
  --class-num 21