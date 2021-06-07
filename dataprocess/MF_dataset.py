# coding:utf-8
import logging
import os
from PIL import Image

from ipdb import set_trace as st

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np

from util.getlog import get_log

logger = get_log()
class MF_dataset(Dataset):

    def __init__(self, data_dir, split, have_label, input_h=480, input_w=640 ,transform=[], client_num=-1, args=None):
        super(MF_dataset, self).__init__()

        self.clients = int(args.client_num)
        assert split in ['train', 'val', 'test'], 'split must be "train"|"val"|"test"'

        f =  open(os.path.join(data_dir, split+'.txt'), 'r') 
        self.temps = [name.strip() for name in f.readlines()]

        if client_num != -1: 
            data_list = []
            i = client_num * 2
            while i < len(self.temps):
                data_list.extend(self.temps[i:i + 2])
                i += self.clients * 2

            self.names = data_list
        else:
            self.names = self.temps
        logger.info(f"Client {client_num} get {len(self.names)} data")
        f.close()

        self.data_dir  = data_dir
        self.split     = split
        self.input_h   = input_h
        self.input_w   = input_w
        self.transform = transform
        self.is_train  = have_label
        self.n_data    = len(self.names)


    def read_image(self, name, folder):
        file_path = os.path.join(self.data_dir, '%s/%s.png' % (folder, name))
        image     = np.asarray(Image.open(file_path)) # (w,h,c)
        image = np.array(image)################
        image.flags.writeable = True
        return image

    def get_train_item(self, index):
        name  = self.names[index]
        image = self.read_image(name, 'images')
        label = self.read_image(name, 'labels')
        
        for func in self.transform:
            image, label = func(image, label)

        image = np.asarray(Image.fromarray(image).resize((self.input_w, self.input_h),resample=Image.NEAREST), dtype=np.float32).transpose((2,0,1))/255
        label = np.asarray(Image.fromarray(label).resize((self.input_w, self.input_h),resample=Image.NEAREST), dtype=np.int64)

        return torch.tensor(image), torch.tensor(label), name

    def get_test_item(self, index):
        self.names = self.names[:19]
        name  = self.names[index]
        image = self.read_image(name, 'images')
        label = self.read_image(name, 'labels')
        
        for func in self.transform:
            image, label = func(image, label)

        image = np.asarray(Image.fromarray(image).resize((self.input_w, self.input_h),resample=Image.NEAREST), dtype=np.float32).transpose((2,0,1))/255
        label = np.asarray(Image.fromarray(label).resize((self.input_w, self.input_h),resample=Image.NEAREST), dtype=np.int64)

        return torch.tensor(image), torch.tensor(label), name
        # name  = self.names[index]
        # image = self.read_image(name, 'images')
        # image = np.asarray(Image.fromarray(image).resize((self.input_w, self.input_h),resample=Image.NEAREST), dtype=np.float32).transpose((2,0,1))/255

        # return torch.tensor(image), name


    def __getitem__(self, index):

        if self.is_train is True:
            return self.get_train_item(index)
        else: 
            return self.get_test_item (index)

    def __len__(self):
        return self.n_data

if __name__ == '__main__':
    data_dir = '../../data/'
    MF_dataset()
