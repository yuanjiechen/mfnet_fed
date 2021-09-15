# coding:utf-8
import numpy as np
import chainer
from PIL import Image
from ipdb import set_trace as st
import ast
import logging
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)
def calculate_accuracy(logits, labels, flag):
    # inputs should be torch.tensor
    predictions = logits.argmax(1)
    no_count = (labels==-1).sum()
    count = ((predictions==labels)*(labels!=-1)).sum()
    acc = count.float() / (labels.numel()-no_count).float()
    return acc


def calculate_result(cf):
    n_class = cf.shape[0]
    conf = np.zeros((n_class,n_class))
    class_acc = np.zeros(n_class)
    IoU = np.zeros(n_class)

    conf[:,0] = cf[:,0]/cf[:,0].sum()

    for pred in range(0, n_class):
        class_acc[pred] = cf[pred,pred] / cf[pred, :].sum()

    for cid in range(1,n_class):
        conf[:,cid] = cf[:,cid]/cf[:,cid].sum()
        IoU[cid]  = cf[cid,cid]/(cf[cid,1:].sum()+cf[1:,cid].sum()-cf[cid,cid])
    overall_acc = np.diag(cf[1:,1:]).sum()/cf[1:,:].sum()
    acc = np.diag(conf)

    return overall_acc, acc, IoU, class_acc


# for visualization
def get_palette():
    unlabelled = [0,0,0]
    car        = [64,0,128]
    person     = [64,64,0]
    bike       = [0,128,192]
    curve      = [0,0,192]
    car_stop   = [128,128,0]
    guardrail  = [64,64,128]
    color_cone = [192,128,128]
    bump       = [192,64,0]
    palette    = np.array([unlabelled,car, person, bike, curve, car_stop, guardrail, color_cone, bump])
    return palette


def visualize(names, predictions):
    palette = get_palette()

    for (i, pred) in enumerate(predictions):
        pred = predictions[i].cpu().numpy()
        img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        for cid in range(1, int(predictions.max())):
            img[pred == cid] = palette[cid]

        img = Image.fromarray(np.uint8(img))
        img.save(names)#[i].replace('.png', '_pred.png'))

def get_logging_config(cfg_file: Union[str, Path]) -> dict:
    """Get the loggging config dictionary from file.

    Parameters
    ----------
    cfg_file : `Union[str, Path]`
        The logging config file.

    Returns
    -------
    `dict`
        The dictionary for the input of dictConfig()
    """
    with open(cfg_file, 'r') as f:
        ret = f.read()
        logging_config = ast.literal_eval(ret)
    
    return logging_config