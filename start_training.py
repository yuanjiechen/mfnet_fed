import argparse
import multiprocessing
from multiprocessing import Process, Pool, Manager
from functools import partial
import logging
import time
from datetime import datetime

from torch.nn import parameter
from functools import partial
import logging.config
from tqdm import tqdm

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


def server_train(obj:server, GPU_queue):
    gpu = GPU_queue.get()
    obj.init_device(gpu_id=gpu)
    while True:
        if not gpu_check(gpu_id=gpu):
            print("No valid GPU")
        else :
            break
        time.sleep(5)  

    obj.split_model()
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
    input()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--client_num",
        type=int,
        default=4,
        required=False,
        help="Decide how many clients in training"
    )
    parser.add_argument(
        "-cp",
        "--check_point",
        type=bool,
        default=False,
        required=False,
        help="Checkpoint avaliable flag"
    )    
    parser.add_argument(
        "-cpd",
        "--check_date",
        type=str,
        default="",
        required=False,
        help="Specify checkpoint,\
            if checkpoint not set, this flag is invalid.\
            format: yyyy-mm-dd-t"
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
        help="Global learning rate"
    )    

    parser.add_argument(
        "-re",
        "--reserve_part",
        type=float,
        default=0.8,
        required=False,
        help="Server and client model aggregation proportion"
    )   

    parser.add_argument(
        "-dt",
        "--data",
        type=str,
        default="../data/",
        required=False,
        help="Server and client model aggregation proportion"
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
        help="result_name"
    )  
    args = parser.parse_args()
    args = arguments(args)
    se = server(args=args)
    args = se.args

    client_num = args.client_num
    client_list = [client(args=args) for i in range(0,client_num)]

    manager = Manager() 

    GPU = manager.Queue()


    GPU.put(0)
    GPU.put(1)
    GPU.put(2)
    GPU.put(3)


    for i in tqdm(range(se.iteration,100)):
        for j in range(len(client_list)):
            #GPU.get()
            p = Process(target=wrapper,args=(client_list[j], GPU, se.iteration, ))
            p.start()

        while True:
            if se.check_num():
                break
            time.sleep(10)

        time.sleep(3)
        if args.reserve_part != 1.0:
            server_train(
                obj=se,
                GPU_queue=GPU
            )
        else:
            logger.info(f"Reserve part : {args.reserve_part} , server not train")


        f = str(args.tag) + ".txt"
        with open(f,"a+") as p:
            gpu = GPU.get()
            se.init_device(gpu_id=gpu)
            p.write(str(se.aggregation())+"\n")
            GPU.put(gpu)