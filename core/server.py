import argparse
from pathlib import Path
import shutil
import math
import sys
from threading import Thread
import logging
import time

sys.path.append("..")

import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
import numpy as np


from util.path_check import checker
from util.connection import connection, sender, recver
from util.parameter import arguments
from util.util import calculate_accuracy, calculate_result
from util.getlog import get_log
from dataprocess.MF_dataset import MF_dataset
from dataprocess.augmentation import RandomFlip, RandomCrop, RandomCropOut, RandomBrightness, RandomNoise
from model.MFnet import MFNet
from model.MF_server import MF_server
from model.MF_RGB_test import MF_RGB_test
from core.server_train import s_train
logger = get_log()
logger.setLevel(logging.INFO)
class server(): 
    def __init__(self, args):
        self.args = args
        self.args,self.path = self.init_path()
        self.init_conn()
        self.args.save_backup(
            outpath=self.path["result"].joinpath("hyperparameter.txt"),
            dic=vars(self.args)
        )
        #print(vars(self.args))
        self.split_name = "split.pt"
        self.model_name = "testmodel.pkl"

        self.model_path = self.path["weight"].joinpath(self.model_name)
        self.split_path = self.path["weight"].joinpath(self.split_name)

        self.recv_path = self.path["train"]
        self.device = torch.device("cpu")

        self.train_loader,\
        self.test_loader = self.init_dataset()
        self.iteration = 1 + self.init_iteration()
        self.init_model()        
        
    def init_path(self):
        path_checker = checker(self.args)
        path = path_checker.server_path()
        path_checker.client_path()
        if path == False:
            print("return fail")
        else :
            return path

    def init_conn(self):
        conn = connection(
            args=self.args,
            send_path=self.path['weight'],
            recv_path=self.path['train']
        )

        conn.establish_conn()
        
    def init_iteration(self):
        path = self.path["result"]
        r = len([file for file in path.rglob("*.pth")])
        logger.info(f"Recover {r} iteration, continue training")
        return r

    def init_dataset(self):
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
            client_num=-1,
            args=self.args
        )

        test_set = MF_dataset(
            data_dir=self.args.data,
            split="val",
            have_label=True,
            client_num=-1, 
            args=self.args     
        )

        train_loader = DataLoader(
            dataset=train_set,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True
        )

        test_loader = DataLoader(
            dataset=test_set,
            batch_size=self.args.test_size,
            shuffle=False,
            num_workers=0,
            drop_last=False            
        )

        logger.info("Dataset initial success")
        logger.info(f"Train_loader length: {train_set.__len__()}")
        logger.info(f"Test_loader length: {test_set.__len__()}")

        return train_loader, test_loader

    def init_device(self, gpu_id):
        if gpu_id < 0:
            self.device = torch.device("cpu")
            return

        self.device = torch.device(f"cuda:{gpu_id}")

        logger.info(f"Iteration {self.iteration} server get GPU {gpu_id} start training")
        return

    def init_model(self):
        i = 1
        if self.args.check_date != "":

            logger.info("Lord checkpoint model")
            while True:
                path = self.path["result"].joinpath(f"round-{i}.pth")
                if path.exists():
                    model = path

                elif not path.exists():
                    break                   

                i += 1

            logger.info(f"Find round {i-1} model")

            shutil.copy(
                src=str(model), 
                dst=str(self.model_path)
            )

        else:
            model = MFNet(n_class=9)
            logger.info("Lord new model")
            torch.save(
                obj=model.state_dict(), 
                f=str(self.model_path)
            )


    def split_model(self):
        fedavg_result = self.recv_path.joinpath(self.model_name)

        if not fedavg_result.exists():
            logger.error(f"Iteration {self.iteration} server model check failed !")
        
        else :
            model_data = {}
            model = MFNet(9)
            split_model = MF_server(n_class=9)
            device=torch.device ("cpu")

            # if self.split_path.exists():
            #     split_model.load_state_dict(
            #         torch.load(
            #             f=str(self.split_path)
            #         )
            #     )
                
            model.load_state_dict(
                torch.load(
                    f=str(fedavg_result),
                    map_location=device
                )
            )

            for name, value in model.state_dict().items():
                model_data[name] = torch.Tensor.float(value)

            for name, value in split_model.state_dict().items():
                if "decode" not in name:
                    split_model.state_dict()[name].copy_(model_data[name])
                else:
                    #print(name)
                    if "conv.weight" in name:
                        origin_size = np.asarray(split_model.state_dict()[name].size()) # [1,2,3,3]
                        get_dim = np.concatenate((-origin_size[:2],origin_size[2:])) # [-1,-2,3,3]
                        split_model.state_dict()[name].copy_(model_data[name][get_dim[0]:, get_dim[1]:, :, :])

                    elif "bn.num_batches_tracked" in name:
                        split_model.state_dict()[name].copy_(model_data[name])

                    else:

                        get_dim = -np.asarray(split_model.state_dict()[name].size())[0]
                        split_model.state_dict()[name].copy_(model_data[name][get_dim:])


        torch.save(
            obj=split_model.state_dict(), 
            f=self.split_path
        )
        return

    def fedavg(self):
        path = self.recv_path
        device=torch.device ("cpu")

        agg_model = MFNet(n_class=9).to(device)

        agg_process = self.recv_path.joinpath(self.model_name)

        model_list = []
        model_data = []

        for fl in path.rglob("*.pth"):
            temp_model = MFNet(9)
            temp_model.load_state_dict(
                torch.load(
                    f=str(fl), 
                    map_location=device                    
                )
            )
            model_list.append(
                temp_model
            )
        logger.info(f"Iteration {self.iteration} recived  {len(model_list)}  model")

        for model in model_list:
            data = {}
            for name, value in model.state_dict().items():
                data[name] = torch.Tensor.float(value)
            model_data.append(data)          

        for name,value in agg_model.state_dict().items():
            # step1 aggregation
            agg_model.state_dict()[name].copy_(torch.mean(torch.stack([data[name] for data in model_data]),dim=0))
            
        torch.save(
            obj=agg_model.state_dict(),
            f=str(agg_process)
        )
        logger.info(f"Iteration {self.iteration} FedAvg model save to {str(agg_process)}")

        # cleaning
        for file in self.recv_path.rglob("*.pth"):
            file.unlink(missing_ok=False)     

        return

    def merger(self):
        logger.info(f"Iteration {self.iteration} start merge c-s model")
        device=torch.device ("cpu")
        trained_path = self.recv_path.joinpath(self.split_name)
        fedavg_path = self.recv_path.joinpath(self.model_name)

        trained_data = {}

        trained_model = MF_server(9)
        agg_model = MFNet(9)

        trained_model.load_state_dict(
            torch.load(
                f=trained_path, 
                map_location=device
            )
        )

        agg_model.load_state_dict(
            torch.load(
                f=fedavg_path,
                map_location=device
            )
        )
        
        for name, value in trained_model.state_dict().items():
            trained_data[name] = torch.Tensor.float(value)


        agg_save = self.model_path
        agg_backup = self.path["result"].joinpath(f"round-{self.iteration}.pth")

        for name,value in trained_data.items():
            if "decode" not in name:
                agg_model.state_dict()[name] = trained_data[name] * (1-self.args.reserve_part) + agg_model.state_dict()[name] * self.args.reserve_part
            else:
                #print(name)
                if "conv.weight" in name:
                    origin_size = np.asarray(trained_data[name].size()) # [1,2,3,3]
                    get_dim = np.concatenate((-origin_size[:2],origin_size[2:])) # [-1,-2,3,3]
                    agg_model.state_dict()[name][get_dim[0]:, get_dim[1]:, :, :] = trained_data[name] * (1-self.args.reserve_part) + agg_model.state_dict()[name][get_dim[0]:, get_dim[1]:, :, :] * self.args.reserve_part
                
                elif "bn.num_batches_tracked" in name:
                    agg_model.state_dict()[name] = trained_data[name] * (1-self.args.reserve_part) + agg_model.state_dict()[name] * self.args.reserve_part
                
                else:
                    get_dim = -np.asarray(trained_data[name].size())[0]
                    agg_model.state_dict()[name][get_dim:] = trained_data[name] * (1-self.args.reserve_part) + agg_model.state_dict()[name][get_dim:] * self.args.reserve_part
        


        for file in self.recv_path.rglob("*.npy"):
            file.unlink(missing_ok=False)

        torch.save(
            obj=agg_model.state_dict(),
            f=str(agg_save)
        )

        torch.save(
            obj=agg_model.state_dict(),
            f=str(agg_backup)
        )
        logger.info(f"Iteration {self.iteration} merged model save to {str(agg_save)}")
        logger.info(f"Iteration {self.iteration} backup model save to {str(agg_backup)}")
        
        result = self.evalute(agg_model)
        self.iteration += 1
        return result

    def baseline(self):
        aggresstion_model = self.recv_path.joinpath(self.model_name)
        agg_backup = self.path["result"].joinpath(f"round-{self.iteration}.pth")
        model = MFNet(9)
        model.load_state_dict(
            torch.load(
                f=aggresstion_model
            )
        )

        # next round
        shutil.copy(
            src=aggresstion_model,
            dst=self.model_path
        )

        # check point
        shutil.copy(
            src=aggresstion_model,
            dst=agg_backup
        )

        result = self.evalute(model)
        self.iteration += 1
        return result
    
    def evalute(self, model):
        cf = np.zeros((9, 9))
        class_acc_all = []
        class_iou_all = []
        class_precision_all = []
        overall_acc_all = []

        loss_avg = 0.
        acc_avg  = 0.
        model.eval()
        time_t = 0.0
        model.to(self.device)
        
        with torch.no_grad():
            for it, (images, labels, names) in enumerate(self.test_loader):
                cf = np.zeros((9, 9))
                images = images.to(self.device)
                labels = labels.to(self.device)
                time_start = time.time()
                logits, _ = model(images)
                time_end = time.time()
                loss = F.cross_entropy(logits, labels)
                acc = calculate_accuracy(logits, labels, True)

                loss_avg += float(loss)
                acc_avg  += float(acc)
                
                #print('|- epo %s/%s. val iter %s/%s. %.2f img/sec loss: %.4f, acc: %.9f' \
                #    % ("1", "100", it+1, len(self.test_loader), 0.0, float(loss), float(acc)))

                predictions = logits.argmax(1)
                for gtcid in range(9): 
                    for pcid in range(9):
                        gt_mask      = labels == gtcid 
                        pred_mask    = predictions == pcid
                        intersection = gt_mask * pred_mask
                        cf[gtcid, pcid] += int(intersection.sum())
                overall_acc, acc, IoU, class_acc = calculate_result(cf)

                overall_acc_all.append(overall_acc)
                class_acc_all.append(class_acc)
                class_iou_all.append(IoU)
                class_precision_all.append(acc)

                time_t += (time_end - time_start)
                # print(IoU)
                #logger.info(f"Iteration {self.iteration} evalute round{it} overall_acc {overall_acc}")
            overall_acc_all = np.nanmean(np.asarray(overall_acc_all))

            
            class_acc_all = np.nanmean(np.asarray(class_acc_all),axis=0)
            class_iou_all = np.nanmean(np.asarray(class_iou_all),axis=0)
            class_precision_all = np.nanmean(np.asarray(class_precision_all),axis=0)

            print('| overall accuracy:', overall_acc_all)
            print('| accuracy:', acc_avg / len(self.test_loader))
            print('| loss:', loss_avg / len(self.test_loader))
            print('| class_acc:', class_acc_all)
            print('| class_IoU:', class_iou_all)
            print('| class_precision:', class_precision_all)
            logger.info(f"Iteration {self.iteration} average overall_acc {overall_acc}")

        result = [overall_acc_all, acc_avg / len(self.test_loader), loss_avg / len(self.test_loader)]

        result.extend(list(class_acc_all))
        result.extend(list(class_iou_all))
        result.extend(list(class_precision_all))
        result.append(time_t / len(self.test_loader))
        #print(result)
        return result

    def training(self):
        model = MF_server(9)
        model.load_state_dict(
            torch.load(
                f = str(self.split_path),
                map_location=self.device
            )
        )

        distill_file = [fl for fl in self.recv_path.rglob("*.npy")]
        distill_array = []
        npy_num = len(distill_file)
        logger.info(f"Iteration {self.iteration} receive {npy_num} distillation file")

        if npy_num != self.args.client_num:
            logger.error(f"Need receive {self.args.client_num} .npy files, \
                but receive {npy_num} files now")
            raise FileNotFoundError("Not enough .npy files !")

        shape = np.array((36, 30, 40))
        if self.args.distill_selection == 1:
            shape = np.array((9, 480, 640))
        for fl in distill_file:
            distill_array.append(
                np.fromfile(
                    file=fl,
                    dtype=np.float32
                ).reshape((shape))#))36, 30, 40
            )
        distill_array = np.mean(
            a=np.asarray(
                a=distill_array,
                dtype=np.float32
            ),
            axis=0,
            keepdims=True,
            dtype=np.float32
        )
        #print(distill_array.shape)
        
        lr = self.args.learning_rate * 0.95 ** (self.iteration - 1)
        logger.info(f"Iteration {self.iteration} learning rate {lr}")

        optim = opt.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        logger.info(f"Server->start training")

        th = Thread(
            target=s_train,
            args=(
                model,
                str(self.recv_path.joinpath(self.split_name)),
                distill_array,
                self.train_loader,
                optim,
                self.args,
                self.device,
            )

        )
        th.setDaemon(True)
        th.start()
        th.join()
        logger.info(f"Iteration {self.iteration} Server->training finish")

        shutil.copy(
            src=str(self.recv_path.joinpath(self.split_name)),
            dst=str(self.split_path)
        )        


    def check_num(self)->bool:
        file_num = len([file for file in self.recv_path.rglob("*.pth")])
        return (file_num == self.args.client_num)

def unit_test():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--client_num", type=int, default=10, required=False)#
    parser.add_argument("-cd", "--check_date", type=str, default="", required=False)#
    #parser.add_argument("-d", "--device", type=str, default="cpu", required=False)
    parser.add_argument("-ml", "--model_name", type=str, default="testmodel.pth", required=False)
    parser.add_argument("-se", "--server_epoch", type=int, default=10, required=False)#
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.01, required=False)#
    parser.add_argument("-op", "--optim", type=str, default="SGD", required=False)
    parser.add_argument("-ls", "--loss_func", type=str, default="cross", required=False)
    parser.add_argument("-re", "--reserve_part", type=float, default=0.8, required=False)#
    parser.add_argument("-bc", "--batch_size", type=int, default=10, required=False)#
    parser.add_argument("-tb", "--test_size", type=int, default=100, required=False)
    parser.add_argument("-dt", "--data", type=str, default="../../data/", required=False)#
    args = parser.parse_args()
    se = server(args=args)
    se.init_conn()
if __name__ == "__main__":
    unit_test()
    
    input()
    input()