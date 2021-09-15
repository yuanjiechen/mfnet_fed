import logging
import sys
from typing import Union

from torch import tensor
sys.path.append("..")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.MFnet import MFNet
from util.getlog import get_log
logger = get_log()
def c_train(model, dst, distill_dst, dataloader,lossfunc, optim, epoch, device):
    epoch_logits = None
    model.to(device)
    for i in range(epoch):
        model.train()
        for it, (images,labels,names) in enumerate(dataloader):
            images = (images).to(device) 
            labels = (labels).to(device)             
            optim.zero_grad()

            logits, distill_logits = model(images)
            loss = lossfunc(logits, labels)


            if i == epoch - 1:
                distill_logits = distill_logits.to("cpu")
                if epoch_logits is None:
                    epoch_logits = distill_logits
                else:
                    epoch_logits = torch.cat(
                        tensors=(epoch_logits, distill_logits),
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