import argparse
import multiprocessing
from multiprocessing import Process, Pool, Manager
from functools import partial
import logging
import time
from datetime import datetime
from pathlib import Path

from functools import partial
import logging.config
from tqdm import tqdm
from pandas.core.frame import DataFrame
import pandas as pd

from core.client import client
from core.server import server
from util.parameter import arguments
from util.dev_check import gpu_check
from util.util import get_logging_config
from util.getlog import get_log
def wrapper(client, GPU, itera):
        gpu = GPU.get()
        counter = 0
        while True:
            if not gpu_check(gpu_id=gpu):
                counter += 1
                print(f"Client {client.num} get GPU {gpu} have not enough memory times {counter} !")
                if counter == 3:
                    GPU.put(gpu)
                    gpu = GPU.get()
                    print(f"Client {client.num} re-get GPU {gpu} !")
                    counter = 0
            else :
                break
            time.sleep(5)  
        client.init_device(gpu)
        client.get_model()
        client.training(itera)
        client.send_model()
        GPU.put(gpu)

def set_result(args:argparse.Namespace):
    distill_dict = {0:"none", 1:"DD", 2:"ED"}
    iid_dict = {0:"non_iid", 1:"iid"}
    result = "_".join([str(args.client_num), str(args.reserve_part), distill_dict[args.distill_selection], str(args.distill_param), iid_dict[args.iid_distribution]])
    
    i = 1
    path = Path(result + f"_{i}.csv")
    while path.exists():
        i += 1
        path = Path(result + f"_{i}.csv")
    args.tag = path
    return

def server_train(obj:server, GPU_queue):
    
    se.split_model()
    gpu = GPU_queue.get()
    obj.init_device(gpu_id=gpu)
    while True:
        if not gpu_check(gpu_id=gpu):
            print("No valid GPU")
        else :
            break
        time.sleep(5)  

    obj.training()
    GPU_queue.put(gpu)

if __name__ == "__main__":
    #LOGGING_CONFIG = get_logging_config('util/logging.conf')
    #logging.config.dictConfig(LOGGING_CONFIG)
    multiprocessing.set_start_method('spawn')

    logging.config.fileConfig(
        fname="./util/logconfig.ini", 
        defaults={
            "logfilename_s": "log/{:%Y-%m-%d}.log".format(datetime.now()), 
            "logfilename_c": "log/{:%Y-%m-%d}client.log".format(datetime.now())
        }
    )
    logger = get_log()
    logger.setLevel(logging.INFO)
    logger.info("New training--------------------\n")
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-c",
        "--client_num",
        type=int,
        default=8,
        required=False,
        help="Decide how many clients in training"
    )
    parser.add_argument(
        "-cpd",
        "--check_date",
        type=str,
        default="",
        required=False,
        help="Specify checkpoint,\
            if checkpoint not set True, this flag is invalid.\
            format: yyyy-mm-dd-t (Deprecated)"
    )   

    parser.add_argument(
        "-se",
        "--server_epoch",
        type=int,
        default=3,
        required=False,
        help="Server training epoch"
    )     

    parser.add_argument(
        "-ce",
        "--client_epoch",
        type=int,
        default=3,
        required=False,
        help="Client training epoch"
    )   

    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=0.01,
        required=False,
        help="Global start learning rate"
    )    

    parser.add_argument(
        "-re",
        "--reserve_part",
        type=float,
        default=0.5,
        required=False,
        help="Server and client model aggregation proportion (alpha)"
    )   

    parser.add_argument(
        "-dt",
        "--data",
        type=str,
        default="/mnt/data5/yuanjie/data/",
        required=False,
        help="Dataset root path"
    )       
    parser.add_argument(
        "-bc",
        "--batch_size",
        type=int,
        default=6,
        required=False,
        help="Global batch size"
    )   
    parser.add_argument(
        "-t",
        "--tag",
        type=str,
        default="result",
        required=False,
        help="Experiment result file name"
    )  
    parser.add_argument(
        "-l",
        "--distill_param",
        type=float,
        default=0.1,
        required=False,
        help="The proportion of distillation loss and label loss (lambda)"
    )
    parser.add_argument(
        "-d",
        "--distill_selection",
        type=int,
        default=2,
        required=False,
        help="Distillation selection: 0-disable 1-DD 2-ED"
    )    
    parser.add_argument(
        "-iid",
        "--iid_distribution",
        type=int,
        default=0,
        required=False,
        help="Specific data distribution is iid=1, non-iid=0,\
            extreme non-iid at client_num=8"
    )
    args = parser.parse_args()
    set_result(args)

    args = arguments(args)
    se = server(args=args)
    args = se.args
    if args.distill_selection == 0:
        args.distill_param = 1.0
        logger.warning("Distillation disabled! ignore the setting of distill_param !")

    client_num = args.client_num
    client_list = [client(args=args) for i in range(0,client_num)]

    manager = Manager() 

    print(args.tag)
    titles = ["overall_acc", "acc", "loss",\
              "class0_acc", "class1_acc", "class2_acc", "class3_acc", "class4_acc", "class5_acc", "class6_acc", "class7_acc", "class8_acc", \
              "class0_IoU", "class1_IoU", "class2_IoU", "class3_IoU", "class4_IoU", "class5_IoU", "class6_IoU", "class7_IoU", "class8_IoU", \
              "class0_precision", "class1_precision", "class2_precision", "class3_precision", "class4_precision", "class5_precision", "class6_precision", "class7_precision", "class8_precision","times"    ]


    GPU = manager.Queue()


    GPU.put(0)
    GPU.put(1)
    GPU.put(2)
    GPU.put(3)

    
    for i in tqdm(range(se.iteration,100)):
        result = []
        print(f"Round{i} start")
        for j in range(len(client_list)):
            #GPU.get()
            p = Process(target=wrapper,args=(client_list[j], GPU, se.iteration, ))
            p.start()

        while True:
            if se.check_num():
                break
            time.sleep(3)

        time.sleep(3)
        se.fedavg()
        if args.reserve_part != 1.0:
            logger.info(f"Reserve part : {args.reserve_part} , advanced mode")
            server_train(
                obj=se,
                GPU_queue=GPU
            )

            gpu = GPU.get()
            se.init_device(gpu_id=gpu)
            result.append(se.merger())
            GPU.put(gpu)
        else:
            logger.info(f"Reserve part : {args.reserve_part} , baseline mode")
            gpu = GPU.get()
            se.init_device(gpu_id=gpu)
            result.append(se.baseline())
            GPU.put(gpu)

        df = DataFrame(result,columns=titles, index=[i])

        if i == 1:
            df.to_csv(str(args.tag), mode="a+")
        else:
            df.to_csv(str(args.tag), mode="a+", header=None)