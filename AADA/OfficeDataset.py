import numpy as np
import os, glob, random, math
from PIL import Image
import torch
import torch.utils.data as data
from torchvision import transforms

class OfficeDataset(data.Dataset):
    def __init__(self, domain, split, transform):
        self.domain = domain
        self.split = split
        self.transform = transform
        if self.domain == 'A':
            self.fpath = 'dataset/office31/amazon/images'
        elif self.domain == 'W':
            self.fpath = 'dataset/office31/webcam/images'
        elif self.domain == 'D':
            self.fpath = 'dataset/office31/dslr/images'

        all_folder = sorted(glob.glob('dataset/office31/dslr/images/*'))
        self.classnames = [f.split('/')[-1] for f in all_folder]
        print(self.classnames)
        self.filepaths = []
        if self.split == 'train':
            for classname in self.classnames:
                all_images = glob.glob(os.path.join(self.fpath,classname,'*.jpg'))
                self.filepaths += all_images
        elif self.split == 'target_train':
            for classname in self.classnames:
                all_images = glob.glob(os.path.join(self.fpath,classname,'*.jpg'))
                num_val = int(len(all_images)*2/3)
                self.filepaths += all_images[:num_val]
        elif self.split == 'target_test':
            for classname in self.classnames:
                all_images = glob.glob(os.path.join(self.fpath,classname,'*.jpg'))
                num_val = int(len(all_images)*2/3)
                self.filepaths += all_images[num_val:]

    def __getitem__(self, index):
        im = Image.open(self.filepaths[index]).convert('RGB')
        class_name = self.filepaths[index].split('/')[-2]
        class_id = self.classnames.index(class_name)
        if self.transform is not None:
            im = self.transform(im)
        return im, class_id

    def __len__(self):
        return len(self.filepaths)
