import argparse
import logging
import math
from model.MFnet import MFNet
from pathlib import Path
from threading import Thread
import sys
import time
from datetime import datetime
sys.path.append("..")

import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
import numpy as np
from dataprocess.MF_dataset import MF_dataset

from util.getlog import get_log
from util.connection import connector, sender, recver
from dataprocess.augmentation import RandomFlip, RandomCrop, RandomCropOut, RandomBrightness, RandomNoise
from core.client_train import c_train
logger = get_log("client")
logger.setLevel(logging.INFO)
class client():
    def __init__(self, server_ip="127.0.0.1", port=8888, home=Path("./cache"), args=None) -> None:
        self.num = -1
        self.sk, self.num = connector(
            ip_addr=server_ip,
            port=args.port
        )
        self.args = args

        self.model_name = "testmodel.pkl"

        self.home = home.joinpath(f"client/client-{self.num}")
        self.model_path = self.home.joinpath(self.model_name)
        self.distill_path = self.home.joinpath("data.npy")

        self.device = torch.device("cpu")
        self.train_loader = self.init_dataloader()

        logger.info(f"client {self.num} initial success")
        #print(f"[{datetime.now()}] Client {self.num} initialization success")
    def init_dataloader(self)->DataLoader:
        train_set = MF_dataset(
            data_dir=self.args.data,
            split="train",
            have_label=True,
            input_h=480,
            input_w=640,
            transform=[
                RandomFlip(prob=0.5),
                RandomCrop(crop_rate=0.1, prob=1.0)
            ],
            client_num=self.num,
            args=self.args
        )

        train_loader = DataLoader(
            dataset=train_set,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True
        )

        return train_loader
    def init_device(self, gpu_id):
        logger.info(f"Client {self.num} get GPU: {gpu_id}")
        self.device = torch.device(f"cuda:{gpu_id}")

    def training(self, itera:int):
        model = MFNet(9)
        model.load_state_dict(
            torch.load(
                f=str(self.model_path),
                map_location=self.device
            )
        )

        model.train()
        lr = self.args.learning_rate * 0.95 ** (itera - 1)
        logger.info(f"Client {self.num} in round {itera} lr = {lr}")
        
        optim = opt.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
            
        logger.info(f"Client {self.num}->start training")
       
        th = Thread(
            target=c_train,
            args=(
                model,
                self.model_path,
                self.distill_path,
                self.train_loader,
                optim,
                self.args,
                self.device,
            )

        )     
        th.setDaemon(True)
        th.start()        
        th.join()
        
        
        logger.info(f"Client {self.num}->training finish")
        return

    def get_model(self):
        self.sk.send("SR".encode(encoding="utf-8"))
        recver(
            sk=self.sk,
            path=self.model_path
        )
        return

    def send_model(self):
        self.sk.send("RE".encode(encoding="utf-8"))
        sender(
            sk=self.sk,
            path=self.model_path
        )

        self.sk.send("DS".encode(encoding="utf-8"))
        sender(
            sk=self.sk,
            path=self.distill_path
        )


def unit_test():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--client_num", type=int, default=10, required=False)
    parser.add_argument("-cd", "--check_date", type=str, default="", required=False)
    #parser.add_argument("-d", "--device", type=str, default="cpu", required=False)
    parser.add_argument("-ml", "--model_name", type=str, default="testmodel.pth", required=False)
    parser.add_argument("-se", "--server_epoch", type=int, default=10, required=False)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.01, required=False)
    parser.add_argument("-op", "--optim", type=str, default="SGD", required=False)
    parser.add_argument("-ls", "--loss_func", type=str, default="cross", required=False)
    parser.add_argument("-re", "--reserve_part", type=float, default=0.8, required=False)
    parser.add_argument("-bc", "--batch_size", type=int, default=10, required=False)
    parser.add_argument("-tb", "--test_size", type=int, default=100, required=False)
    parser.add_argument("-dt", "--data", type=str, default="../../data/", required=False)
    args = parser.parse_args()
    cl = client(args=args)
    cl.get_model()
    cl.send_model()
    
    print(cl.num)


if __name__ == "__main__":
    unit_test()