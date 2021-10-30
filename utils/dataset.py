import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import os
from os.path import join

class CustomImageDataset(Dataset):

    def default_loader(path):
        return Image.open(path).convert('RGB')

    def __init__(self, file_list, dir, transform=None, target_transform=None, loader=default_loader):
        img_labels = []
        img_dir = join(dir,'training_images')

        for line in file_list:
            line = line.strip('\n')
            words = line.split()
            img_path = os.path.join(img_dir, words[0])
            label = words[1].split(".")[0]
            img_labels.append((img_path, int(label)-1))
        self.img_labels = img_labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path, label = self.img_labels[idx]
        image = self.loader(img_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label