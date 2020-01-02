from torch.utils.data import DataLoader
from .dataset.voc import VOCDataset
from .dataset.coco import COCODataset

def data_loader(args):
    datalist = args.train_list

    if args.dataset == 'PascalVOC':
        dataset = VOCDataset(
            root=args.data,
            gt_root=args.gt_root,
            datalist=datalist,
        )
    elif args.dataset == 'COCO':
        dataset = COCODataset(
            root=args.data,
            gt_root=args.gt_root,
            datalist=datalist,
        )
    else:
        raise Exception("No matching dataset.")

    dataset_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers
    )
    return dataset_loader

