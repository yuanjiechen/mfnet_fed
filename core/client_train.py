import logging
import sys
from typing import Union
import copy
import time

from torch import tensor
sys.path.append("..")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.MFnet import MFNet
from util.getlog import get_log
logger = get_log()

def fedprox(model, global_model, prox_term):
    if prox_term == 0:
        return 0
    prox = 0.0
    for w, w_g in zip(model.parameters(), global_model.parameters()):
        prox += (w-w_g).norm(2)

    return prox

def c_train(model, dst, distill_dst, dataloader, optim, args, device):
    epoch_logits = []
    global_model = copy.deepcopy(model)

    global_model.to(device)
    model.to(device)

    epoch = args.client_epoch
    prox_term = args.prox
    
    lossfunc = nn.CrossEntropyLoss()
    for i in range(epoch):
        model.train()
        for it, (images,labels,names) in enumerate(dataloader):
            images = (images).to(device) 
            labels = (labels).to(device)             
            optim.zero_grad()

            logits, distill_logits = model(images)
            prox = fedprox(model, global_model, prox_term)

            loss = lossfunc(logits, labels) + prox * prox_term
            #print("Prox value: ", prox * prox_term)
            if args.distill_selection == 1:
                distill_logits = logits

            if i == epoch - 1:
                distill_logits = distill_logits.to("cpu")
                epoch_logits.append(distill_logits)

            loss.backward()
            optim.step()
    epoch_logits = torch.cat(
        tensors=epoch_logits, 
        dim=0
    )

    avg_logits = torch.mean(
        input=epoch_logits,
        dtype=torch.float32,
        dim=0
    ).detach().numpy()

    torch.save(
        obj=model.state_dict(),
        f=str(dst)
    )

    avg_logits.tofile(distill_dst)
