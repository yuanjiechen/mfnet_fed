import logging
import sys
from typing import Union

from torch import tensor
sys.path.append("..")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.MF_server import MF_server
from util.getlog import get_log

logger = get_log()

def s_train(model, dst, distill_array, dataloader, optim, args, device):
    lossfunc = nn.CrossEntropyLoss()
    distill_loss = nn.MSELoss()

    model.to(device)
    ld = args.distill_param
    epoch = args.server_epoch
    for i in range(epoch):
        model.train()
        for it, (images,labels,names) in enumerate(dataloader):
            images = (images).to(device) 
            labels = (labels).to(device) 

            optim.zero_grad()
            logits, distill_logits = model(images)
            if args.distill_selection == 1:
                distill_logits = logits 
                distill_loss = nn.KLDivLoss()
            loss = lossfunc(logits, labels)

            teacher = torch.tensor(
                data=distill_array,
                dtype=torch.float32
            ).to(device)#[1, 9, 480, 640]
            #print(teacher.size())
            #print(" A ")


            student = torch.mean(
                input=distill_logits,
                dim=0,
                keepdim=True,
                dtype=torch.float32
            )
            #print(student.size())
            if args.distill_selection == 1:
                teacher = F.softmax(
                    input=teacher,
                    dim=1,
                    dtype=torch.float32
                )

                student = F.log_softmax(
                    input=student,
                    dim=1,
                    dtype=torch.float32
                )
            #print(teacher.size(), student.size())
            
            loss2 = distill_loss(student, teacher)
            #print(loss.item(),loss2.item())
            #logger.info(f"KLD loss: {loss2.item()}")
            loss = ld * loss + (1-ld) * loss2

            loss.backward()
            optim.step()

    torch.save(
        obj=model.state_dict(),
        f=str(dst)
    )

