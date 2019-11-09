import os
import cv2
import pylab
import pickle
import numpy as np
import scipy.ndimage as nd

import torch
from torch.utils.data import Dataset
MODE = ['train', 'val', 'test']
IMG_FOLDER_NAME = "JPEGImages"
IMAGE_MEAN_VALUE = [104.0, 117.0, 123.0]


class VOCDataset(Dataset):

    def __init__(self, root=None, gt_root=None, datalist=None, resize_size=321, num_category=21, mode='train'):
        if mode not in MODE:
            raise Exception("Mode has to be one of ", MODE)
        self.root = root
        self.resize_size = resize_size

        if mode == 'train':
            self.image_names = [img_gt_name.split()[0][:-4]
                                for img_gt_name in open(datalist).read().splitlines()]
            self.gt_names = None
            self.label_list = load_image_label_list_from_npy(self.image_names)
        elif mode == 'val':
            self.image_names = [img_gt_name.split()[0][-15:-4]
                                for img_gt_name in open(datalist).read().splitlines()]
            self.gt_names = [img_gt_name.split(' ')[1][1:]
                             for img_gt_name in open(datalist).read().splitlines()]
        elif mode == 'test':
            self.image_names = [img_gt_name[-15:-4]
                                for img_gt_name in open(datalist).read().splitlines()]
        else:
            raise Exception("No matching mode {}".format(mode))

        self.cue_data = pickle.load(open(gt_root, 'rb'))
        self.mode = mode
        self.num_category = num_category

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        image_path = os.path.join(self.root, IMG_FOLDER_NAME, name + '.jpg')

        # Load Image
        img = pylab.imread(image_path)
        img = preprocess(img, self.resize_size, IMAGE_MEAN_VALUE).astype(np.float32)

        if self.mode == 'train':
            img = torch.from_numpy(img)
            label, gt_map = self.get_seed(idx)
            label = torch.from_numpy(label)
            gt_map = torch.from_numpy(gt_map.astype(np.float32))
            return img, label, gt_map

        elif self.mode == 'val':
            img = torch.from_numpy(img)
            gt_name = self.gt_names[idx]
            return img, gt_name, name

        elif self.mode == 'test':
            img = torch.from_numpy(img)
            return img, None, name

        else:
            raise Exception("No matching mode {}".format(self.mode))

    def get_seed(self, id_slice):
        label_seed = np.zeros((1, 1, self.num_category))
        label_seed[0, 0, self.cue_data["%s_labels" % id_slice]] = 1.
        cues = np.zeros([21, 41, 41])
        cues_i = self.cue_data["%s_cues" % id_slice]
        cues[cues_i[0], cues_i[1], cues_i[2]] = 1.
        return label_seed.astype(np.float32),cues.astype(np.float32)


def load_image_label_list_from_npy(img_name_list):
    cls_labels_dict = np.load('datalist/PascalVOC/cls_labels.npy', allow_pickle=True).item()
    return [cls_labels_dict[img_name] for img_name in img_name_list]


def load_img_cue_name_list(dataset_path):
    img_cue_name_list = open(dataset_path).read().splitlines()
    img_name_list = [img_cue_name.split(' ')[0][-15:-4] for img_cue_name in img_cue_name_list]
    cue_name_list = [img_cue_name.split(' ')[1].strip() for img_cue_name in img_cue_name_list]

    return img_name_list, cue_name_list


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
    return image