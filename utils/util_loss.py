import numpy as np
from scipy.ndimage import zoom
from krahenbuhl2013 import CRF

import torch
import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')
MIN_PROB = 1e-4


def softmax_layer(preds):
    preds = preds
    pred_max, _ = torch.max(preds, dim=1, keepdim=True)
    pred_exp = torch.exp(preds - pred_max.clone().detach())
    probs = pred_exp / torch.sum(pred_exp, dim=1, keepdim=True) + MIN_PROB
    probs = probs / torch.sum(probs, dim=1, keepdim=True)
    return probs


def seed_loss_layer(probs, labels):
    count = torch.sum(labels, dim=[1, 2, 3], keepdim=True)
    loss_balanced = -torch.mean(torch.sum(labels * torch.log(probs), dim=[1, 2, 3], keepdim=True) / count)
    return loss_balanced


def expand_loss_layer(probs_tmp, stat_inp):

    stat = stat_inp[:, :, :, 1:]

    probs_bg = probs_tmp[:, 0, :, :]
    probs = probs_tmp[:, 1:, :, :]

    probs_max, _ = torch.max(torch.max(probs, dim=3)[0], dim=2)

    q_fg = 0.996
    probs_sort, _ = torch.sort(probs.contiguous().view(-1, 20, 41 * 41), dim=2)
    weights = probs_sort.new_tensor([q_fg ** i for i in range(41 * 41 -1, -1, -1)])[None, None, :]
    z_fg = torch.sum(weights)
    # weight = ..
    probs_mean = torch.sum((probs_sort * weights) / z_fg, dim=2)

    q_bg = 0.999
    probs_bg_sort, _ = torch.sort(probs_bg.contiguous().view(-1, 41 * 41), dim=1)
    weights_bg = probs_sort.new_tensor([q_bg ** i for i in range(41 * 41 -1, -1, -1)])[None, :]
    z_bg = torch.sum(weights_bg)
    # weights_bg = ..
    probs_bg_mean = torch.sum((probs_bg_sort * weights_bg) / z_bg, dim=1)

    stat_2d = (stat[:, 0, 0, :] > 0.5).float()
    loss_1 = -torch.mean(torch.sum((stat_2d * torch.log(probs_mean) / torch.sum(stat_2d, dim=1, keepdim=True)), dim=1))
    loss_2 = -torch.mean(torch.sum(((1 - stat_2d) * torch.log(1 - probs_max) / torch.sum(1 - stat_2d, dim=1, keepdim=True)), dim=1))
    loss_3 = -torch.mean(torch.log(probs_bg_mean))

    loss = loss_1 + loss_2 + loss_3
    return loss


def crf_layer(fc8_SEC, images, iternum):
    unary = np.transpose(np.array(fc8_SEC.cpu().clone().data), [0, 2, 3, 1])
    mean_pixel = np.array([104.0, 117.0, 123.0])
    im = images.cpu().data
    im = zoom(im, (1, 1, 41 / im.shape[2], 41 / im.shape[3]), order=1)

    im = im + mean_pixel[None, :, None, None]
    im = np.transpose(np.round(im), [0, 2, 3, 1])

    N = unary.shape[0]

    result = np.zeros(unary.shape)

    for i in range(N):
        result[i] = CRF(im[i], unary[i], maxiter=iternum, scale_factor=12.0)
    result = np.transpose(result, [0, 3, 1, 2])
    result[result < MIN_PROB] = MIN_PROB
    result = result / np.sum(result, axis=1, keepdims=True)

    return np.log(result)


def constrain_loss_layer(probs, probs_smooth_log):
    probs_smooth = torch.exp(probs.new_tensor(probs_smooth_log, requires_grad=True))
    loss = torch.mean(torch.sum(probs_smooth * torch.log(probs_smooth / probs), dim=1))

    return loss
















