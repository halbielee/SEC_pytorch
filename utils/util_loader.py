from torch.utils.data import DataLoader
from .dataset.voc import VOCDataset

IMAGE_MEAN_VALUE = [104., 117., 123.]

def data_loader(args, mode='train'):
    if mode == 'train':
        datalist = args.train_list
        shuffle = True
    elif mode == 'val':
        datalist = args.val_list
        shuffle = False
    elif mode == 'test':
        datalist = args.test_list
        shuffle = False
    else:
        raise Exception("Error: There is no {} mode.".format(mode))

    dataset = VOCDataset(
        root=args.data,
        gt_root=args.gt_root,
        datalist=datalist,
        mode=mode,
    )
    dataset_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers
    )
    return dataset_loader

