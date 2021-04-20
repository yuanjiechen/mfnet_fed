from pickle import TRUE
import socket
from threading import Thread
import sys
import os
import struct
import time
import calendar
import shutil
import math
import gc

import torch.nn as nn
import torch.optim as opt

import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np

from pathlib import Path
import torch.nn.functional as F
from torch.autograd import Variable
# from MLP import MLP
# from CNN import CNN
#from dataload import generate_dataset
from parameter import pm
#from pytorchtools import EarlyStopping
from util.MF_dataset import MF_dataset
from util.util import calculate_accuracy, calculate_result
from util.augmentation import RandomFlip, RandomCrop, RandomCropOut, RandomBrightness, RandomNoise
from model.MFNet import MFNet#, SegNet
device = torch.device("cpu")

#from process.bar import Bar
# server 发送接收模型，整合模型，确定client数量，划分资料
# client 请求连接，训练

def transfer_data(dest,modelpath):
    path = modelpath
    try:
        fhead = struct.pack('512sl', os.path.basename(path).encode(encoding="utf-8"), os.stat(path).st_size)
    except IOError as msg:
        print("Model not found!")
        raise
    try:
        dest.sendall(fhead)
        file = open(path,'rb')
        while True:
            data = file.read(1024)
            if not data:
                #print("Model sended")
                break
            dest.sendall(data)
        
        file.close()
    except (socket.error, IOError) as msg:
        print(msg)
        raise
    return

def recv_model(region,path):
    fileinfo_size = struct.calcsize("512sl")
    try:

        buffer = region.recv(fileinfo_size,socket.MSG_WAITALL)
        filename,filesize = struct.unpack("512sl",buffer)
        filename = filename.strip(b'\00')
        filename = filename.decode(encoding="utf-8")
        

        file_recv = open(path,"wb")
        recv_size = 0
        while not filesize == recv_size:
            if filesize - recv_size > 1024:
                data = region.recv(1024,socket.MSG_WAITALL)
                recv_size = recv_size + len(data)
            else :
                data = region.recv(filesize - recv_size)
                recv_size = filesize
            
            file_recv.write(data)  
        file_recv.close()
        #print("Receive success")

    except Exception as msg:#(socket.error,struct.error,IOError) as msg:
        print(msg)
        print("Recv model error!")
        raise

def init_path(client_home, num):

    if os.path.exists(client_home) == False:
        os.makedirs(client_home)

    client_data_path = client_home + "/client-" + str(num)

    if os.path.exists(client_data_path) == False:
        os.makedirs(client_data_path)


    return client_data_path

class server_():
    def __init__(self,modelpath):
        self.filepath = modelpath
        self.parameter = pm().getdata()
        self.init_port = 8888
        self.sk = None
        self.init_connect()
        self.clientlist = []
        #self.clientaddr = []
        self.model = self.init_model(1)
       # self.pm = {name : value for name ,value in self.model.named_parameters()}
        self.recv_position = self.init_recvpath() + "/"
        self.agg_model = None

        
        self.batch = int(self.parameter["test_batch"])
        self.epoch = int(self.parameter["epoch"])
        self.client_num = int(self.parameter["clients"])
        self.test_dataloader = self.init_dataloader()#self.wordmodel)
        
        self.iteration = 1
        self.complete_num = 0
        self.recv_num = 0
        self.train_num = 0
        self.init_num = 0

        self.logits_backup = None
    def init_connect(self):
        self.sk = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        while True:
            if self.check_port(self.init_port) == False :
                self.init_port = self.init_port + 1
            else:
                break
        try:
            self.sk.bind(('0.0.0.0',self.init_port))
            self.sk.listen(10)
            
            print("Initial success",", port : ",self.init_port)
        except socket.error as msg:
            print(msg,"Some error,try again")

        th = Thread(target=self.command_process)
        th.setDaemon(True)
        th.start()

        th2 = Thread(target=self.process_inst)
        th2.setDaemon(True)
        th2.start()        
    def init_model(self, new):
        if new == 1:
            new_model = MFNet(9)#MLP(28*28, 250, 100, 10)
            torch.save(new_model.state_dict(), self.filepath)
        return None#torch.load(self.filepath)

    def init_datapath(self, client_home , client_num):
        init_path(client_home, client_num)

    def init_recvpath(self):
        path = str(os.path.dirname(__file__)) + "/train/"
        nowdate = str(time.strftime("%Y-%m-%d", time.localtime()))
        times = 1
        while os.path.exists(path + nowdate + "-" + str(times)) == True:
            times = times + 1
        os.makedirs(path + nowdate + "-" + str(times))
        return path + nowdate + "-" + str(times)
    def init_dataloader(self):#, wordmodel):
        #ds = open(self.parameter["test_path"], encoding="utf-8").readlines()
        #trans = torchvision.transforms.ToTensor()
        # test_data = torchvision.datasets.FashionMNIST(root="~/test/MNIST", train=False,transform=trans,download=False)
        # test_loader = DataLoader(test_data,batch_size=100,shuffle=False)
        test_dataset = MF_dataset('../data/', 'val', have_label=True)
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=100,
            shuffle=True
        )
        return test_loader

    def check_port(self,port):
        sk = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        try:
            sk.connect(('0.0.0.0',port))
            sk.close()
            return False
        except:
            return True

    def process_inst(self):
        while True:
            conn,addr = self.sk.accept()
            #print("Client connect success")
            self.clientlist.append(conn)
            #self.clientaddr.append(addr)
            th = Thread(target=self.message_pass,args=(conn,))
            #th.setDaemon(True)
            th.start()
    def message_pass(self,client):
        try :
            client.setblocking(True)
            client_num = self.clientlist.index(client)
            #client.sendall("Connect success".encode(encoding='utf-8'))
            client.sendall((str(client_num)).encode(encoding='utf-8'))
        except socket.error as msg:
            print(msg)
            client.close()
            self.clientlist.remove(client)
            raise
            return

        while True:
            try:
                data = client.recv(2).decode(encoding='utf-8',errors="ignore")

            except ConnectionResetError as msg:
                print(msg)
                client.close()
                self.clientlist.remove(client)
                return

            if len(data) == 0:
                client.close()
                self.clientlist.remove(client)
                return
            elif data == "0":
                continue
            elif data == 'IN':
                self.init_num += 1
                uncomplete = self.client_num - self.init_num
                print("Client init :[" + "||" * int(15 * self.init_num / self.client_num ) + "__" * int(1 - 15 * self.init_num / self.client_num ) + ']', end="\r")
                if(uncomplete == 0):
                    print("\n")
            elif data == 'SR':
                try:
                    transfer_data(client,self.filepath)

                except:
                    print("Send model failed")
                    client.close()
                    self.clientlist.remove(client)
                    raise
                    return

            elif data == 'RE':
                try:
                    recv_model(client,self.recv_position + "client-" + str(self.clientlist.index(client)) + ".pkl")
                except Exception as msg:
                    print(msg)
                    client.close()
                    self.clientlist.remove(client)
                    raise
                    return   
                     
            elif data == "33":
                uncomplete = 9 - self.complete_num
                self.complete_num = self.complete_num + 1
                #print("Recv model :" + "<<" * self.complete_num + "__" * uncomplete, end="\r")
                #if(uncomplete == 0):
                #    print("\n")
                
                
            elif data == "SD": # data_trans
                self.recv_num = self.recv_num + 1
                uncomplete = self.client_num - self.recv_num

                print("Round :" + str(self.iteration) + " Send model :[" + ">>" * int(15 * self.recv_num / self.client_num ) + "__" * int(15 * (1 - self.recv_num / self.client_num )) + ']', end="\r")
                if(uncomplete == 0):
                    print("\nRound :" + str(self.iteration) + " Training :" + '[' + "__" * 15 + ']', end='\r')
                

            elif data == "TR":
                self.train_num = self.train_num + 1
                uncomplete = self.client_num * self.epoch - self.train_num

                print("Round :" + str(self.iteration) + " Training :" + '[' + "##" * int(15 * self.train_num / (self.client_num * self.epoch)) + "__" * int(15 * (1 - self.train_num / (self.client_num * self.epoch))) + ']', end="\r")
                if(uncomplete == 0):
                    print("\n")
            else:
                print(data)

    def command_process(self):

        while True:
            print("\nMenu")
            print("1: client num")
            print("2: close socket")
            print("3: aggerate model")
            print("4: complete num\n")
            command = input()
            if command == "1":
                print("client:  ",len(self.clientlist))
            elif command == "2":
                self.sk.close()
                for client in self.clientlist:
                    client.close()
                os._exit(0)
            elif command == "3":
                self.aggerate(self.recv_position)
            elif command == "4":
                print(self.complete_num)
            else :
                print("please re input")

    def aggerate(self,path):
        #print("start aggreate !")
        self.complete_num = 0
        self.recv_num = 0
        self.train_num = 0
        model_count = 0
        model_list = []
        model_data = []
        #
        # remove_list = []
        for dp,_,fs in os.walk(path):
            for fl in fs:
                try:
                    if fl != 'result.txt':
                        temp_model = MFNet(9)
                        model_count += 1
                        # remove_list.append(fl)
                        temp_model.load_state_dict(torch.load(dp + "/" + fl, map_location=device))
                        model_list.append(temp_model)
                        #break
                        # model_list[-1].to(device)
                except BaseException as e:
                    print(e)
                    raise
                    
            break
        #print("Load model success !")
        for client in model_list:
            data = {}
            for name,value in client.state_dict().items():
                data[name] = torch.Tensor.float(value) 
            model_data.append(data)
        
        agg_model = MFNet(9)#MLP(28*28, 250, 100, 10)
        agg_model = agg_model.to(device)
        #print(data['conv5_rgb.conv1_left.bn.bias'])
        for name,value in agg_model.state_dict().items():
            agg_model.state_dict()[name].copy_(torch.mean(torch.stack([data[name] for data in model_data]),dim=0))


        print("aggerate success !")
        #torch.save(agg_model,self.filepath)
        torch.save(agg_model.state_dict(), self.filepath)
        result = self.evaluate(agg_model)#, self.wordmodel)
        #print("Round : " + str(self.iteration) + "  Recvd model : " + str(model_count))
        #self.iteration = self.iteration + 1
        # for i in range(len(remove_list)):
        #     os.remove(os.path.join(path ,remove_list[i]))
        return result, model_count
    '''
    def evaluate(self, model):
        model.eval()
        correct = 0
        for image,label in self.test_dataloader:
            image = image.view(image.size(0),-1)
            image = image.to(device)
            label = label.to(device)
            out = model.forward(image)
            _,maxdata = torch.max(out,1)
            correct = correct + (maxdata==label).sum()
        acc = correct.cpu().numpy() / 100
        print ("Round : " + str(self.iteration) + " Global accuracy : " + str(acc))
        return acc
    '''
    def evaluate(self, model):
        cf = np.zeros((9, 9))
        loss_avg = 0.
        acc_avg  = 0.
        model.eval()
        # for name,value in model.named_parameters():
        #     print(name)
        #     print(value.data)
        #     break
        #print(len(self.test_dataloader))
        correct = 0
        acc_total = 0
        loss_avg = 0.
        acc_avg  = 0.
        
        w = open("/home/yuanjie/test.txt", "w+")
        with torch.no_grad():
            for it, (images, labels, names) in enumerate(self.test_dataloader):
                images = Variable(images)
                labels = Variable(labels)

                images = images.to(device)
                labels = labels.to(device)
                
                
                logits = model(images)
                
                #loss = F.cross_entropy(logits, labels)
                #loss.backward()
                loss = F.cross_entropy(logits, labels)
                acc = calculate_accuracy(logits, labels, True)
                loss_avg += float(loss)
                acc_avg  += float(acc)
                
                print('|- epo %s/%s. val iter %s/%s. %.2f img/sec loss: %.4f, acc: %.9f' \
                    % ("1", "100", it+1, len(self.test_dataloader), 0.0, float(loss), float(acc)))


                predictions = logits.argmax(1)
                for gtcid in range(9): 
                    for pcid in range(9):
                        gt_mask      = labels == gtcid 
                        pred_mask    = predictions == pcid
                        intersection = gt_mask * pred_mask
                        cf[gtcid, pcid] += int(intersection.sum())
                overall_acc, acc, IoU = calculate_result(cf)
                print('| overall accuracy:', overall_acc)
                print('| accuracy of each class:', acc)
                print('| class accuracy avg:', acc.mean())
                print('| IoU:', IoU)
                print('| class IoU avg:', IoU.mean())
                break
        return overall_acc


class client_():
    def __init__(self):
        self.num = -1
        
        self.sk = None
        self.init_conn()
        self.device = self.init_device(0)
        self.parameters = pm().getdata()
        #self.args = args

        self.recv_path = init_path(self.parameters["client_home"], self.num) + "/"     #接收model和存data的路径
        self.server_model_path = self.recv_path + self.parameters["server_send_name"]
        self.time_path = self.recv_path + self.parameters["time_record"]

        self.loss_fn = self.parameters["loss_fn"]
        self.optim =  self.parameters["optim"]
        self.lr = float(self.parameters["lerning_rate"])
        self.epoch = int(self.parameters["epoch"])
        self.batch = int(self.parameters["batch_size"])
        self.early_stop = int(self.parameters["early_stop_param"])
        self.sets_class = int(self.parameters["sets"])
        self.iid = False

        
        self.data_rough = int(self.parameters["rough"])
        self.train_loader, self.test_loader = self.init_dataloader()

        self.recv_model = 'testmodel.pkl'
        self.train_model = 'model_train.pkl'

        self.sk.send("IN".encode(encoding="utf-8"))

    def init_conn(self):
        self.sk = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        try:
            self.sk.connect(("140.114.89.42",8889))
            self.sk.setblocking(True)

            self.num = int(self.sk.recv(10).decode(encoding='utf-8'))

            return
        except Exception as identifier:
            print("Init failed Check service!")
            raise
    
    def init_device(self, device_num:int):
        # if self.num >= 0 and self.num < 10:
        #     return torch.device("cuda:0")
        # elif self.num >= 10 and self.num < 20:
        #     return torch.device("cuda:1")
        # elif self.num >=20 and self.num < 30:
        #     return torch.device("cuda:2")
        # else :
        #     return torch.device("cuda:3")
        self.device = torch.device("cuda:{}".format(device_num))
         

    def init_dataloader(self):#->tuple(DataLoader,):
        #train_set, test_set = generate_dataset(iid=self.iid, sets=self.sets_class, clients_num=self.num)
        train_set = MF_dataset("../data/", 'train', have_label=True, transform=[
            RandomFlip(prob=0.5),
            RandomCrop(crop_rate=0.1, prob=1.0), 
        ], client_num=self.num)
        train_loader = DataLoader(train_set,batch_size=self.batch,shuffle=True, num_workers=0)
        #test_loader = DataLoader(test_set,batch_size=self.batch,shuffle=False)

        return train_loader, None#test_loader
        
    def record_time(self, t_str):
        f = open(self.time_path, "a+", encoding="utf-8")
        f.write(t_str)
        f.close()
        
    def training(self, model, itera):
        time_start = time.time()

        #early_stop = EarlyStopping(patience=10,verbose=True)    
        #print("Client :" + str(self.num) + " start training")
        lr = self.lr * 0.95 ** (itera-1)
        if model == None:
            model = MFNet(9)#torch.load(self.recv_path + self.recv_model, map_location=self.device)
            model.load_state_dict(torch.load(self.recv_path + self.recv_model, map_location=self.device))

        if self.loss_fn == "mse":
            lossfunction = nn.MSELoss()
        elif self.loss_fn == "cro":
            lossfunction = nn.CrossEntropyLoss()
        if self.optim == "adam":
            optim = opt.Adam(params=model.parameters(),lr=self.lr / math.pow(itera, 1/3))
        elif self.optim == "SGD":
            optim = opt.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        #print("Client at :"+ str(self.num) + str(self.device) + " start training")

        model = model.to(self.device)
        loss_avg = 0.
        acc_avg  = 0.
        model.train()
        epoch = self.epoch

        for i in range(epoch):
            model.train()
            for it, (images,labels,names) in enumerate(self.train_loader):
                images = Variable(images).to(self.device) 
                labels = Variable(labels).to(self.device) 
                #print(labels)
                optim.zero_grad()
                logits = model(images)
                #print(logits.shape)
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                optim.step()

                acc = calculate_accuracy(logits, labels)
                loss_avg += float(loss)
                acc_avg  += float(acc)
                #print('|- epo %s/%s. train iter %s/%s. %.2f img/sec loss: %.4f, acc: %.4f' \
                #% ("1", "100", it+1, len(self.train_loader), (it+1)*0.0, float(loss), float(acc)))
            #print(acc)
            self.sk.send("TR".encode(encoding="utf-8"))
            

            #vaild_losses = []
            #print ("client   : ",self.num,"  :" ,train_loss,"   " )#, valid_loss)

        torch.save(model.state_dict(), self.recv_path + self.train_model)
        #torch.save(model.state_dict(), "/home/yuanjie/MFNet-pytorch-master/weights/MFNet/testmodel.pkl")

        time_end = time.time()
        self.record_time("Training time :" + str(time_end - time_start)+ "  " + str(self.parameters['rough'] + "\n"))
        
        #del optim

        #model.to(torch.device('cpu'))
        #del (model)
        #cuda.close()
        

    def get_model(self):
        self.sk.send("SR".encode(encoding="utf-8"))
        recv_model(self.sk, self.server_model_path)
        
        
        self.sk.send("SD".encode(encoding="utf-8"))
        return

    def send_model(self, itera):
        
        self.sk.send("RE".encode(encoding="utf-8"))
        transfer_data(self.sk, str(Path(self.recv_path).joinpath(self.train_model)))
        self.sk.send("33".encode(encoding="utf-8"))

        #print(f"{self.round}   oooooooo")
        return


        


# 服务器初始化连接,等待用户连接,指令：请求训练/请求回收模型，确认模型状态，若用户数量大于**，向用户发送模型与随机码，等待用户训练完成，回收模型，校验随机码，/其他资料，关闭连接