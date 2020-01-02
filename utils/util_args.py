# -*- coding: utf-8 -*-
import argparse
import network as models

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch PascalVOC Training')
    parser.add_argument('--name', type=str, default='test_case')
    parser.add_argument('--evaluate', action='store_true', help='evaluate mode')
    parser.add_argument('--arch', default='resnet18', choices=model_names, help='model choice')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--pretrained', action='store_true', help='use pre-trained model')
    parser.add_argument('--resume', default='', type=str, help='checkpoint path to resume')
    parser.add_argument('--snapshot', default=1000, type=int, help='snapshot point')

    # hyperparamter
    parser.add_argument('--max-iter', type=int, default=8000, help='number of total iteration to run')
    parser.add_argument('--lr-decay', type=int, default=2000, help='Reducing lr frequency')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    parser.add_argument('--batch-size', default=64, type=int,  help='mini-batch size')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--wd', default=1e-4, type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--nest', action='store_true', help='nestrov for optimizer')

    # path
    parser.add_argument('--data', default='/workspace', help='path to dataset')
    parser.add_argument('--dataset', type=str, default='PASCAL', )
    parser.add_argument('--gt-root', type=str, default='datalist/PascalVOC/localization_cues-sal.pickle')
    parser.add_argument('--train-list', type=str, default='datalist/PascalVOC/input_list.txt')

    # data transform
    parser.add_argument('--resize-size', type=int, default=321, help='input resize size')
    parser.add_argument('--crop-size', type=int, default=321, help='input crop size')

    parser.add_argument('--debug', default=False, action='store_true')

    args = parser.parse_args()

    if args.dataset == 'PascalVOC':
        args.train_list = 'datalist/PascalVOC/input_list.txt'

    elif args.dataset == 'COCO':
        if args.debug:
            args.train_list = 'datalist/COCO/input_list_debug.txt'
        else:
            args.train_list = 'datalist/COCO/input_list.txt'

    return args
