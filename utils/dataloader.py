import logging
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from os.path import join

from .dataset import CustomImageDataset

logger = logging.getLogger(__name__)

def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    train_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                transforms.RandomCrop((448, 448)),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomRotation(30),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    val_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                transforms.CenterCrop((448, 448)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    data_root = args.data_root
    annotations_file = join(data_root, 'training_labels.txt')

    f = open(annotations_file)
    filename = f.readlines()
    train_set_size = int(len(filename) * 0.8)
    valid_set_size = len(filename) - train_set_size

    train_set = CustomImageDataset(file_list=filename[:train_set_size], dir=data_root,transform=train_transform)
    val_set = CustomImageDataset(file_list=filename[train_set_size:], dir=data_root,transform=val_transform)                        

    train_loader = DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=args.eval_batch_size, shuffle=True, num_workers=4)

    return train_loader, val_loader
