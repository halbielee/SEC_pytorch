# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.optim
import network as models
from tensorboardX import SummaryWriter

from tqdm import tqdm
from utils.util import get_parameters, save_checkpoint, load_model, adjust_learning_rate
from utils.util_args import get_args
from utils.util_loader import data_loader
from utils.util_loss import \
    softmax_layer, seed_loss_layer, expand_loss_layer, crf_layer, constrain_loss_layer
torch.set_num_threads(4)
import torchvision.utils as vutils
import torch.nn.functional as F

def print_grad(x):
    print(x.requires_grad, x.is_leaf)

def main():
    args = get_args()
    log_folder = os.path.join('train_log', args.name)
    writer = SummaryWriter(log_folder)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # number of classes for each dataset.
    if args.dataset == 'PascalVOC':
        num_classes = 21
    else:
        raise Exception("No dataset named {}.".format(args.dataset))

    # Select Model & Method
    model = models.__dict__[args.arch](num_classes=num_classes)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    # Optimizer
    optimizer = torch.optim.SGD([
        {'params': get_parameters(model, bias=False, final=False), 'lr':args.lr, 'weight_decay': args.wd},
        {'params': get_parameters(model, bias=True, final=False), 'lr':args.lr * 2, 'weight_decay': 0},
        {'params': get_parameters(model, bias=False, final=True), 'lr':args.lr * 10, 'weight_decay': args.wd},
        {'params': get_parameters(model, bias=True, final=True), 'lr':args.lr * 20, 'weight_decay': 0}
    ], momentum=args.momentum)

    if args.resume:
        model = load_model(model, args.resume)

    train_loader = data_loader(args)

    data_iter = iter(train_loader)
    train_t = tqdm(range(args.max_iter))
    model.train()
    for global_iter in train_t:
        try:
            images, target, gt_map = next(data_iter)
        except:
            data_iter = iter(data_loader(args))
            images, target, gt_map = next(data_iter)


        if args.gpu is not None:
            images = images.cuda(args.gpu)
            gt_map = gt_map.cuda(args.gpu)
            target = target.cuda(args.gpu)

        output = model(images)

        fc8_SEC_softmax = softmax_layer(output)
        loss_s = seed_loss_layer(fc8_SEC_softmax, gt_map)
        loss_e = expand_loss_layer(fc8_SEC_softmax, target)
        fc8_SEC_CRF_log = crf_layer(output, images, iternum=10)
        loss_c = constrain_loss_layer(fc8_SEC_softmax, fc8_SEC_CRF_log)

        loss = loss_s + loss_e + loss_c

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # writer add_scalars
        writer.add_scalars(args.name, {'loss_s': loss_s, 'loss_e': loss_e, 'loss_c': loss_c}, global_iter)

        with torch.no_grad():
            if global_iter % 10 == 0:
                # writer add_images (origin, output, gt)

                # image_mean_value = [.485, .456, .406]
                # image_std_value = [.229, .224, .225]
                # image_mean_value = torch.reshape(torch.tensor(image_mean_value), (1, 3, 1, 1)).cuda(args.gpu)
                # image_std_value = torch.reshape(torch.tensor(image_std_value), (1, 3, 1, 1)).cuda(args.gpu)
                # origin = images.clone().detach() * image_std_value + image_mean_value
                origin = images.clone().detach() + torch.tensor([104., 117., 123.]).reshape(1, 3, 1, 1).cuda(args.gpu)

                size = (100, 100)
                origin = F.interpolate(origin, size=size)
                origins = vutils.make_grid(origin, nrow=15, padding=2)

                outputs = F.interpolate(output, size=size)
                _, outputs = torch.max(outputs, dim=1)
                outputs = outputs.unsqueeze(1)
                outputs = vutils.make_grid(outputs, nrow=15, padding=2, normalize=True, scale_each=True).float()

                gt_maps = F.interpolate(gt_map, size=size)
                _, gt_maps = torch.max(gt_maps, dim=1)
                gt_maps = gt_maps.unsqueeze(1)
                gt_maps = vutils.make_grid(gt_maps, nrow=15, padding=2, normalize=True, scale_each=True).float()

                # gt_maps = F.interpolate(gt_map.unsqueeze(1).float(), size=size)
                # gt_maps = vutils.make_grid(gt_maps, nrow=15, padding=2, normalize=True, scale_each=True).float()

                grid_image = torch.cat((origins, outputs, gt_maps), dim=1)
                writer.add_image(args.name, grid_image, global_iter)


        description = '[{0:4d}/{1:4d}] loss: {2} s: {3} e: {4} c: {5}'.\
            format(global_iter+1, args.max_iter, loss, loss_s, loss_e, loss_c)
        train_t.set_description(desc=description)

        # save snapshot
        if global_iter % args.snapshot == 0:
            save_checkpoint(model.state_dict(), log_folder, 'checkpoint_%d.pth.tar' % global_iter)

        # lr decay
        if global_iter % args.lr_decay == 0:
            args.lr = args.lr * 0.1
            optimizer = adjust_learning_rate(optimizer, args.lr)

    print("Training is over...")
    save_checkpoint(model.state_dict(), log_folder, 'last_checkpoint.pth.tar')



if __name__ == '__main__':
    main()