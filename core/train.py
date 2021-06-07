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
from model.MF_server import MF_server
from model.MF_RGB_test import MF_RGB_test
from util.getlog import get_log
logger = get_log()
def train(mark, model, dst, data_dst, distill_array, dataloader ,lossfunc, optim, lr, epoch, device):
    epoch_logits = None
    avg_logits = None
    distill_loss = nn.KLDivLoss(
        log_target=False
    )
    model.to(device)
    model.train()
    for i in range(epoch):
        model.train()
        for it, (images,labels,names) in enumerate(dataloader):
            images = (images).to(device) 
            labels = (labels).to(device) 

            optim.zero_grad()
            logits = model(images)

            loss = lossfunc(logits, labels)

            
            if i == epoch - 1 and mark == "client":
                logits = logits.cpu()
                if epoch_logits is None:
                    epoch_logits = logits
                else:
                    epoch_logits = torch.cat(
                        tensors=(epoch_logits, logits),
                        dim=0
                    )
            
            
            if mark == "server":
                logits = logits.to(device)


                logits = torch.mean( #[batch, 9, 480, 640] → [9, 480, 640]
                    input=logits,
                    dtype=torch.float32,
                    dim=0
                )

                logits = torch.unsqueeze(
                    input = logits,
                    dim = 0
                )

                student = F.log_softmax(
                    input=logits,
                    dim=1,
                    dtype=torch.float32
                ).to(device)

                teacher = F.softmax(
                    input=torch.unsqueeze(
                        input=torch.tensor(
                            data=distill_array, # from client
                            dtype=torch.float32,
                        ),
                        dim=0
                    ),
                    dim=1,
                    dtype=torch.float32
                ).to(device)
                # print(teacher.size())
                # print(student.size())
                
                loss2 = distill_loss(student, teacher)

                loss = 0.5 * loss + 0.5 * loss2

            loss.backward()
            optim.step()    


    if mark == "client":
        avg_logits = torch.mean( # float32
            input=epoch_logits,
            dtype=torch.float32,
            dim=0
        ).detach().numpy()
        
            
    torch.save(
        obj=model.state_dict(),
        f=str(dst)
    )

    if data_dst is not None:
        avg_logits.tofile(
            file=data_dst
        )
