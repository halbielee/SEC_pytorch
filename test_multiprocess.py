import os
import cv2
import pylab
import warnings
import argparse
import numpy as np
import scipy.ndimage as nd
from multiprocessing import Process

import torch
import network as models
from utils.util import load_model
import krahenbuhl2013

warnings.filterwarnings('ignore', '.*output shape of zoom.*')

IMAGE_MEAN_VALUE = [104.0, 117.0, 123.0]

def parser_args():
    parser = argparse.ArgumentParser(description='Get segmentation prediction')
    parser.add_argument("--image-list", type=str, help="Path to image")
    parser.add_argument("--image-path", type=str, help="Path to image")
    parser.add_argument("--arch", type=str, default='deeplab_large_fov', help="Model type")
    parser.add_argument("--trained", type=str, help="Model weights")
    parser.add_argument("--pred-path", type=str, help="Output png file name", default='')
    parser.add_argument("--smooth", action='store_true', help="Apply postprocessing")
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]', default=0, type=int)
    parser.add_argument('--split-size', type=int, default=5)
    parser.add_argument('--num-gpu', type=int, default=1)

    args = parser.parse_args()
    return args


def preprocess(image, size, mean_pixel):
    image = np.array(image)

    image = nd.zoom(image.astype('float32'),
                    (size / float(image.shape[0]), size / float(image.shape[1]), 1.0),
                    order=1)

    # RGB to BGR
    image = image[:, :, [2, 1, 0]]
    image = image - np.array(mean_pixel)

    # BGR to RGB
    image = image.transpose([2, 0, 1])
    return np.expand_dims(image, 0)


def predict_mask(image_file, model, smooth, gpu_id):
    im = pylab.imread(image_file)

    image = torch.from_numpy(preprocess(im, 321, IMAGE_MEAN_VALUE).astype(np.float32))
    image = image.cuda(gpu_id)
    output = model(image)
    scores = output.reshape(21, 41, 41).detach().cpu().numpy().transpose(1, 2, 0)
    d1, d2 = float(im.shape[0]), float(im.shape[1])

    scores_exp = np.exp(scores - np.max(scores, axis=2, keepdims=True))
    probs = scores_exp / np.sum(scores_exp, axis=2, keepdims=True)
    probs = nd.zoom(probs, (d1 / probs.shape[0], d2 / probs.shape[1], 1.0), order=1)

    eps = 0.00001
    # eps = 0.00001 * 50.
    probs[probs < eps] = eps

    if smooth:
        result = np.argmax(krahenbuhl2013.CRF(im, np.log(probs), scale_factor=1.0), axis=2)
    else:
        result = np.argmax(probs, axis=2)

    return result


def save_mask_multiprocess(num, data_size):
    process_id = os.getpid()
    print('process {} starts...'.format(process_id))

    if args.num_gpu == 1:
        gpu_id = args.gpu_id
    elif args.num_gpu == 2:
        if num >= data_size // args.num_gpu:
            gpu_id = args.gpu_id + 0
        else:
            gpu_id = args.gpu_id + 1
    elif args.num_gpu == 4:
        if num >= data_size // args.num_gpu * 3:
            gpu_id = args.gpu_id + 0
        elif num >= data_size // args.num_gpu * 2:
            gpu_id = args.gpu_id + 1
        elif num >= data_size // args.num_gpu * 1:
            gpu_id = args.gpu_id + 2
        else:
            gpu_id = args.gpu_id + 3
    else:
        raise Exception("ERROR")

    model = models.__dict__[args.arch](21)
    model = model.cuda(gpu_id)
    model = load_model(model, args.trained)
    model.eval()

    if num == data_size - 1:
        sub_image_ids = image_ids[num * len(image_ids) // data_size:]
    else:
        sub_image_ids = image_ids[num * len(image_ids) // data_size: (num + 1) * len(image_ids) // data_size]
    if num == 0:
        print(len(sub_image_ids), 'images per each process...')

    for idx, img_id in enumerate(sub_image_ids):
        if num == 0 and idx % 10 == 0:
            print("[{0} * {3}]/[{1} * {3}] : {2} is done.".format(idx, len(sub_image_ids), img_id, args.split_size))
        image_file = os.path.join(image_path, img_id + '.jpg')
        mask = predict_mask(image_file, model, args.smooth, gpu_id)
        save_path = os.path.join(args.pred_path, img_id + '.png')
        cv2.imwrite(save_path, mask)


if __name__ == "__main__":
    args = parser_args()
    image_ids = [i.strip() for i in open(args.image_list) if not i.strip() == '']
    image_path = os.path.join(args.image_path, 'JPEGImages')

    if args.pred_path and (not os.path.isdir(args.pred_path)):
        os.makedirs(args.pred_path)

    split_size = args.split_size * args.num_gpu
    numbers = range(split_size)
    processes = []
    for index, number in enumerate(numbers):
        proc = Process(target=save_mask_multiprocess, args=(number, split_size,))
        processes.append(proc)
        proc.start()

    for proc in processes:
        proc.join()