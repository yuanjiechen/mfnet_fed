import multiprocessing
import queue
from threading import Thread
import argparse
import time
import socket
import shutil
import argparse
from multiprocessing import Process, Pool, Manager
import os
from functools import partial

import torch
import torch.nn as nn
from tqdm import tqdm

from server_ import server_, client_
#from dataload import *
from parameter import *
#from MLP import *
from pre_work import clear




# 启动服务器，加载参数，设定训练参数，生成客户，进行训练流程，整合模型，测试模型，记录结果，下一轮训练
#parameters = pm().getdata()
#generate_rand_data(parameters, "/home/yuanjie/pp20/k-mean-data/")

def start_client(client, GPU, itera):#temp, client_1, client_2, GPU):
    try:
        # server send common words
        # send data type
        #client = client_1.get()
        gpu = GPU.get()
        time.sleep(5)
        client.init_device(gpu)
        client.get_model()
        client.training(None, itera)
        client.send_model(itera)

        #client_2.put(client)
        GPU.put(gpu)

        client.sk.close()
        
    except Exception as e:
        print(e)
        raise


if __name__ == '__main__':
    #multiprocessing.set_start_method('spawn')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--client_num', type=int, default=20, required=False, help='Client number, max 40')
    parser.add_argument('-i', '--iid', type=bool, default=False, required=False, help='Generate iid or non-iid data')
    parser.add_argument('-s', '--sets', type=int, default=3, required=False, help='Data per client')
    args = parser.parse_args()
    #dev_check(20)
    clear()

    parameters = pm().getdata()

    server = server_("/home/yuanjie/mf_fed/weight/testmodel.pkl")  
    # model = torch.load("/home/yuanjie/mf_fed/MLP/weight/testmodel.pkl", map_location=torch.device("cuda:0"))
    # result = server.evaluate(model)  
    client_num = int(parameters["clients"])     

    client_list = [client_() for i in range(0,client_num)]
    
    manager = Manager() 
    '''
    client_1 = manager.Queue() #full
    client_2 = manager.Queue()
    GPU = manager.Queue()

    

    for client in client_list:
        client_1.put(client)
    '''
    GPU = manager.Queue()

    GPU.put(0)
    GPU.put(1)
    GPU.put(2)
    GPU.put(3)

    #temp = list(range(0, int(parameters["clients"])))
    #with Pool(40) as pool:

    for i in tqdm(range(0 ,200)):  
        try:
            '''
            if not client_1.empty():
                pfunc = partial(start_client, client_1=client_1, client_2=client_2, GPU=GPU)
                #for client in range(int(parameters["clients"])):
                list(tqdm(pool.imap_unordered(pfunc,temp), total=len(temp)))

            elif not client_2.empty():
                pfunc = partial(start_client, client_1=client_2, client_2=client_1, GPU=GPU)
                #for client in range(int(parameters["clients"])):
                list(tqdm(pool.imap_unordered(pfunc,temp), total=len(temp)))
            else :
                print("else      ss ")

            '''
            for j in range(len(client_list)):
                #GPU.get()
                p = Process(target=start_client,args=(client_list[j], GPU, server.iteration, ))
                p.start()

            
                #p.join()
        except Exception as e:
            print(e)
            raise

        
        while True:
            #os.system("nvidia-smi")
            time.sleep(10)

            if server.complete_num == client_num:
                
                result_file = open( server.recv_position + "result.txt", "a+", encoding="utf-8")
                result = server.aggerate(server.recv_position)
                result_file.write(str(result) + "\n")
                result_file.close()
                server.iteration += 1
                break
        
        server.complete_num = 0
    server.sk.close()
    input()

    