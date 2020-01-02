import os
import cv2
import pickle
import numpy as np
import scipy.ndimage as nd
import torch
from torch.utils.data import Dataset

# Color mean in order RGB
IMAGE_MEAN_VALUE = [123.0, 104.0, 117.0]


class COCODataset(Dataset):

    def __init__(self, root=None, gt_root=None, datalist=None, resize_size=321, num_category=81):
        self.root = root
        self.resize_size = resize_size

        self.image_names = [img_gt_name.strip().split()[0][:-4] for img_gt_name in open(datalist).readlines()]
        self.id_list = [img_gt_name.strip().split()[1] for img_gt_name in open(datalist).readlines()]
        self.cue_data = pickle.load(open(gt_root, 'rb'))
        self.num_category = num_category

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        image_path = os.path.join(self.root, 'train2014', name + '.jpg')

        # Load Image
        img = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        img = preprocess(img, self.resize_size, IMAGE_MEAN_VALUE).astype(np.float32)

        img = torch.from_numpy(img)
        label, gt_map = self.get_seed(idx)
        label = torch.from_numpy(label)
        gt_map = torch.from_numpy(gt_map.astype(np.float32))
        return img, label, gt_map

    def get_seed(self, id_slice):
        id_slice = self.id_list[id_slice]
        label_seed = np.zeros((1, 1, self.num_category))
        label_seed[0, 0, self.cue_data["%s_labels" % id_slice]] = 1.
        cues = np.zeros([81, 41, 41])
        cues_i = self.cue_data["%s_cues" % id_slice]
        cues[cues_i[0], cues_i[1], cues_i[2]] = 1.
        return label_seed.astype(np.float32),cues.astype(np.float32)


def preprocess(image, size, mean_pixel):
    image = np.array(image)

    image = nd.zoom(image.astype('float32'),
                    (size / float(image.shape[0]), size / float(image.shape[1]), 1.0), order=1)
    image = image - np.array(mean_pixel)
    image = image.transpose([2, 0, 1])
    return image