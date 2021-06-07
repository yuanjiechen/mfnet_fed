import argparse
from pathlib import Path
import shutil
import math
import sys
from threading import Thread
import logging


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
from core.train import train

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

        self.model_path = self.path["weight"].joinpath(self.args.model_name)
        self.split_path = self.path["weight"].joinpath(self.split_name)
        self.recv_path = self.path["train"]
        self.device = torch.device("cpu")

        self.train_loader,\
        self.test_loader = self.init_dataset()
        self.iteration = 1 + self.init_iteration()
        self.init_model()
        self.split_model()
        
        
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
        if self.args.check_point == 1:

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

        elif self.args.check_point == 0:
            model = MFNet(n_class=9)
            logger.info("Lord new model")
            torch.save(
                obj=model.state_dict(), 
                f=str(self.model_path)
            )


    def split_model(self):
        if not self.model_path.exists():
            logger.info(f"iteration-{self.iteration}-check model failed !")

        else :
            model_data = {}
            model = MFNet(9)
            model.load_state_dict(
                torch.load(
                    f=str(self.model_path)
                )
            )

            split_model = MF_server(n_class=9)
            if self.split_path.exists():
                split_model.load_state_dict(
                    torch.load(
                        f=str(self.split_path)
                    )
                )
                logger.info(f"iteration {self.iteration} lord split model last round")
            
            for name, value in model.state_dict().items():
                model_data[name] = torch.Tensor.float(value)
            
            for name, value in split_model.state_dict().items():
                if "decode" not in name:
                    split_model.state_dict()[name].copy_(model_data[name])

            torch.save(
                obj=split_model.state_dict(), 
                f=self.split_path
            )
        return

    def aggregation(self)->str:
        path = self.recv_path
        device=torch.device ("cpu")
        trained_path = self.path["train"].joinpath(self.split_name)
        trained_data = {}

        trained_model = MF_server(9)
        no_agg = False
        
        try:
            trained_model.load_state_dict(
                torch.load(
                    f=trained_path, 
                    map_location=device
                )
            )
        except FileNotFoundError:
            if self.args.reserve_part == 1.0:
                logger.warn("Server model not found")
                no_agg = True
            else:
                logger.error(f"Server model not found, reserve part {self.args.reserve_part} not 1.0")
                raise FileNotFoundError


        agg_model = MFNet(n_class=9).to(device)
        agg_save = self.model_path
        agg_backup = self.path["result"].joinpath(f"round-{self.iteration}.pth")


        logger.info(f"iteration {self.iteration} save to {str(agg_save)}")
        logger.info(f"iteration {self.iteration} save backup to {str(agg_backup)}")

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

        for model in model_list:
            data = {}
            for name, value in model.state_dict().items():
                data[name] = torch.Tensor.float(value)
            model_data.append(data)
        
        logger.info(f"iteration {self.iteration} recived  {len(model_list)}  model")
    
        # extract data 
        for name,value in trained_model.state_dict().items():
            trained_data[name] =  torch.Tensor.float(value)

        logger.info(f"iteration {self.iteration} reserve_part {self.args.reserve_part} ")
        logger.info(f"Aggregation {not no_agg} ")
        for name,value in agg_model.state_dict().items():
            # step1 aggregation
            agg_model.state_dict()[name].copy_(torch.mean(torch.stack([data[name] for data in model_data]),dim=0))
            # step2 aggregation
            if name in trained_data and "decode" not in name and no_agg == False:
                agg_model.state_dict()[name] = trained_data[name] * (1-self.args.reserve_part) + agg_model.state_dict()[name] * self.args.reserve_part

        torch.save(
            obj=agg_model.state_dict(),
            f=str(agg_save)
        )

        torch.save(
            obj=agg_model.state_dict(),
            f=str(agg_backup)
        )

        # cleaning
        try:
            for file in self.recv_path.rglob("*.pth"):
                file.unlink(missing_ok=False)

            for file in self.recv_path.rglob("*.npy"):
                file.unlink(missing_ok=False)            


            trained_path.unlink(missing_ok=True)

        except FileExistsError:
            logger.error("Model receive not complete ! Abort ")
            raise FileExistsError

        result = self.evalute(agg_model)
        self.iteration += 1
        return result
    
    def evalute(self, model):
        cf = np.zeros((9, 9))
        loss_avg = 0.
        acc_avg  = 0.
        model.eval()

        loss_avg = 0.
        acc_avg  = 0.
        model.to(self.device)
        overall_acc_all = 0.0
        with torch.no_grad():
            for it, (images, labels, names) in enumerate(self.test_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
        
                logits = model(images)
                
                loss = F.cross_entropy(logits, labels)
                acc = calculate_accuracy(logits, labels, True)
                loss_avg += float(loss)
                acc_avg  += float(acc)
                
                print('|- epo %s/%s. val iter %s/%s. %.2f img/sec loss: %.4f, acc: %.9f' \
                    % ("1", "100", it+1, len(self.test_loader), 0.0, float(loss), float(acc)))


                predictions = logits.argmax(1)
                for gtcid in range(9): 
                    for pcid in range(9):
                        gt_mask      = labels == gtcid 
                        pred_mask    = predictions == pcid
                        intersection = gt_mask * pred_mask
                        cf[gtcid, pcid] += int(intersection.sum())
                overall_acc, acc, IoU = calculate_result(cf)
                print('| overall accuracy:', overall_acc)
                # print('| accuracy of each class:', acc)
                # print('| class accuracy avg:', acc.mean())
                # print('| IoU:', IoU)
                # print('| class IoU avg:', IoU.mean())
                overall_acc_all += overall_acc
                #logger.info(f"Iteration {self.iteration} evalute round{it} overall_acc {overall_acc}")
            
            print('| overall accuracy:', overall_acc_all / len(self.test_loader))
            logger.info(f"Iteration {self.iteration} average overall_acc {overall_acc_all / len(self.test_loader)}")
        
        return overall_acc_all / len(self.test_loader)

    def training(self):
        model = MF_server(9)
        model.load_state_dict(
            torch.load(
                f = str(self.split_path),
                map_location=self.device
            )
        )

        distill_file = [fl for fl in self.path["train"].rglob("*.npy")]
        distill_array = []
        npy_num = len(distill_file)
        logger.info(f"Iteration {self.iteration} receive {npy_num} distillation file")

        if npy_num != self.args.client_num:
            logger.error(f"Need receive {self.args.client_num} .npy files, \
                but receive {npy_num} files now")
            raise FileNotFoundError("Not enough .npy files !")

        for fl in distill_file:
            distill_array.append(
                np.fromfile(
                    file=fl,
                    dtype=np.float32
                ).reshape((9, 480, 640))
            )
        distill_array = np.mean(
            a=np.asarray(
                a=distill_array,
                dtype=np.float32
            ),
            axis=0,
            dtype=np.float32
        )

        
        lr = self.args.learning_rate * 0.95 ** (self.iteration - 1)
        logger.info(f"Iteration {self.iteration} learning rate {lr}")

        if self.args.loss_func == "cross":
            lossfunc = nn.CrossEntropyLoss()
        elif self.args.loss_func == "mse":
            lossfunc = nn.MSELoss()

        if self.args.optim == "adam":
            optim = opt.Adam(params=model.parameters(),lr=self.args.lr / math.pow(self.iteration, 1/3))
        elif self.args.optim == "SGD":
            optim = opt.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        logger.info(f"Server->start training")

        th = Thread(
            target=train,
            args=(
                "server",
                model,
                str(self.path["train"].joinpath(self.split_name)),
                None,
                distill_array,
                self.train_loader,
                lossfunc,
                optim,
                lr,
                self.args.server_epoch,
                self.device,
            )

        )
        th.setDaemon(True)
        th.start()
        th.join()
        logger.info(f"Iteration {self.iteration} Server->training finish")
        
        # torch.save(
        #     obj=model.state_dict(),
        #     f=str(self.path["train"].joinpath(self.split_name))
        # )
        shutil.copy(
            src=str(self.path["train"].joinpath(self.split_name)),
            dst=str(self.split_path)
        )
        
        
        logger.info(f"Save split model to {self.split_path}")


    def check_num(self)->bool:
        file_num = len([file for file in self.recv_path.rglob("*.pth")])
        # if self.args.reserve_part != 1.0:
        #     server_train = self.path["train"].joinpath(self.split_name).exists()
        # else :
        #     #logger.info(f"Server model existing check ignored, reserve part : {self.args.reserve_part}")
        #     server_train = True

        if file_num == self.args.client_num: # and server_train:
            return True
        else:
            return False

def unit_test():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--client_num", type=int, default=10, required=False)#
    parser.add_argument("-ck", "--check_point", type=bool, default=False, required=False)#
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