import os
import torch
import torch.nn as nn

def get_parameters(model, bias=False,final=False):
    if final:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                if m.out_channels == 21:
                    if bias:
                        yield m.bias
                    else:
                        yield m.weight
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                if not m.out_channels == 21:
                    if bias:
                        yield m.bias
                    else:
                        yield m.weight


def save_checkpoint(state, save_dir, filename='checkpoint.pth.tar'):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    torch.save(state, os.path.join(save_dir, filename))


def load_model(model, model_path):
    if os.path.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
    checkpoint = torch.load(model_path)
    origin_state_dict = model.state_dict()
    model.load_state_dict(checkpoint, strict=False)
    return model


def adjust_learning_rate(optimizer, lr):
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr']=2*lr
    optimizer.param_groups[2]['lr']=10*lr
    optimizer.param_groups[3]['lr']=20*lr
    return optimizer