import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import scipy.ndimage as nd


class SeedingLoss(nn.Module):

    def __init__(self):
        super(SeedingLoss, self).__init__()

    def forward(self, seg_map, cues):
        count = cues.sum()
        loss = -(cues * seg_map.log()).sum() / count

        return loss


class balanced_seed_loss(nn.Module):

    def __init__(self):
        super(balanced_seed_loss, self).__init__()
        self.loss1 = nn.CrossEntropyLoss()
        self.loss2 = nn.CrossEntropyLoss()
    def forward(self, softmax, cues):

        count_bg = torch.sum(cues[:,:1], dim=(1,2,3), keepdim=True)
        loss_bg = torch.mean(torch.sum(cues[:,:1]* torch.log(softmax[:,:1]), dim=(1,2,3), keepdim=True) / (count_bg + 1e-8))
        count_fg = torch.sum(cues[:,1:], dim=(1,2,3), keepdim=True)
        loss_fg = torch.mean(torch.sum(cues[:,1:]* torch.log(softmax[:,1:]), dim=(1,2,3), keepdim=True) / (count_fg + 1e-8))
        # loss1 = self.loss1(softmax[:,:1], cues[:,:1])
        # loss2 = self.loss2(softmax[:,1:], cues[:,1:])
        return loss_fg + loss_bg

class contrain_loss(nn.Module):
    def  __init__(self):
        super(contrain_loss, self).__init__()

    def forward(self, softmax, crf):
        probs_smooth = torch.exp(crf)
        loss = torch.mean(torch.sum(probs_smooth * torch.log(probs_smooth / (softmax + 1e-8)+1e-8), dim=1))
        return loss
