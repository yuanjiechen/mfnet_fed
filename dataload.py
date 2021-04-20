import random
import time

import torch.nn as nn
import torchvision
import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
from gensim.models import  Word2Vec

import os
import logging
from parameter import pm

class Getdataset(Dataset):
    def __init__(self, dataset:Dataset, indexs:np.array):
        self.dataset = dataset
        self.idxs = [int(i) for i in indexs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self,idx):

        img, label = self.dataset[self.idxs[idx]]
        return torch.as_tensor(img), torch.as_tensor(label)


def mnist_non_iid(sets:int, clients_num:int)->np.array:
    param = pm().getdata()
    user_num = int(param['clients'])
    trans = torchvision.transforms.ToTensor()
    train_data = torchvision.datasets.FashionMNIST(root="~/test/MNIST", train=True,transform=trans,download=False)
    test_data = torchvision.datasets.FashionMNIST(root="~/test/MNIST", train=False,transform=trans,download=False)

    num_set, num_img = 200, 300
    idx_set = [i for i in range(num_set)] #200 sets

    user_dict = {i:np.array([]) for i in range(user_num)}

    total_idx = np.arange(num_set * num_img)
    labels = train_data.targets.numpy()

    idx_label = np.vstack((total_idx,labels)) # [[idx1,idx2,......], [label1, label2,......]]
    #print(idx_label)
    idx_label = idx_label[:, idx_label[1, :].argsort()]  # 先对label做sort, 再取对应index的值
    total_idx = idx_label[0, :]

    for i in range(user_num):
        rand_set = set(np.random.choice(idx_set,sets,replace=False))
        idx_set = list(set(idx_set) - rand_set)
        for rand in rand_set:
            user_dict[i] = np.concatenate((user_dict[i], total_idx[rand * num_img : (rand + 1) * num_img]), axis=0)

    

    #print(idx_label)
    return user_dict[clients_num]

def mnist_iid(sets:int, clients_num:int):
    param = pm().getdata()
    user_num = int(param['clients'])
    trans = torchvision.transforms.ToTensor()
    train_data = torchvision.datasets.FashionMNIST(root="~/test/MNIST", train=True,transform=trans,download=False)
    test_data = torchvision.datasets.FashionMNIST(root="~/test/MNIST", train=False,transform=trans,download=False)

    num_set, num_img = 1000, 60
    idx_set = [i for i in range(num_set)] #1000 sets

    user_dict = {i:np.array([]) for i in range(user_num)}

    total_idx = np.arange(num_set * num_img)
    labels = train_data.targets.numpy()

    idx_label = np.vstack((total_idx,labels)) # [[idx1,idx2,......], [label1, label2,......]]
    #print(idx_label)
    idx_label = idx_label[:, idx_label[1, :].argsort()]  # 先对label做sort, 再取对应index的值
    total_idx = idx_label[0, :]

    for i in range(user_num):
        #for j in range(10):
            # rand_set = set(np.random.choice(idx_set[j * (100 - i): (j + 1) * (100 - i)], size=1, replace=False))
            # print(rand_set)
            #idx_set = list(set(idx_set) - rand_set)
        rand_set = [idx_set[i] for i in range(clients_num, len(idx_set), 100)]
        for rand in rand_set:
            user_dict[i] = np.concatenate((user_dict[i], total_idx[rand * num_img : (rand + 1) * num_img]), axis=0)


    return user_dict[clients_num]

def generate_dataset(iid:bool, sets:int, clients_num:int):
    trans = torchvision.transforms.ToTensor()
    train_data = torchvision.datasets.FashionMNIST(root="~/test/MNIST", train=True,transform=trans,download=False)
    test_data = torchvision.datasets.FashionMNIST(root="~/test/MNIST", train=False,transform=trans,download=False)
    
    if iid == True:
        train_data = Getdataset(train_data, mnist_non_iid(sets,clients_num))
        return train_data, test_data

    elif iid == False:
        train_data = Getdataset(train_data, mnist_iid(sets,clients_num))
        return train_data, test_data
    

#mnist_non_iid()