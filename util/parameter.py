import argparse
import csv
import logging
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.INFO)
class arguments():
    def __init__(self,args=None,path=Path(__file__).absolute().parent.joinpath("hyperparameter.txt")) -> None:
        self.path = path
        self.default = self.get_default()
        logger.info(f"Lord parameters from {path}")
        if "client_num" not in self.default:
            self.client_num = args.client_num
            self.check_date = args.check_date
            self.tag = args.tag
            self.distill_selection = args.distill_selection
            self.server_epoch = args.server_epoch
            self.client_epoch = args.client_epoch
            self.learning_rate = args.learning_rate
            self.reserve_part = args.reserve_part
            self.distill_param = args.distill_param
            self.batch_size = args.batch_size
            self.data = args.data
            self.iid = args.iid_distribution
        else :
            self.client_num = int(self.default["client_num"])
            self.check_date = args.check_date
            self.tag = self.default["tag"]
            self.distill_selection = int(self.default["distill_selection"])
            self.server_epoch = int(self.default["server_epoch"])
            self.client_epoch = int(self.default["client_epoch"])
            self.learning_rate = float(self.default["learning_rate"])
            self.distill_param = float(self.default["distill_param"])
            self.reserve_part = float(self.default["reserve_part"])
            self.batch_size = int(self.default["batch_size"])
            self.data = self.default["data"]   
            self.iid = int(self.default["iid"])

        self.loss_func = self.default["loss_func"]
        self.optim = self.default["optim"]
        self.test_size = int(self.default["test_size"])



    def get_default(self)->dict:
        csv_ = pd.read_csv(str(self.path), encoding="utf-8",header=None)
        data_dict = {}
        origin_data = csv_.to_dict(orient="records")

        for data in origin_data:
            data_dict[data[0]] = data[1]

        return data_dict

    def save_backup(self, outpath, dic:dict):
        dic_ = dic.copy()
        del dic_["path"]
        del dic_["default"]
        del dic_["check_date"]
        pd.DataFrame(list(dic_.items())).to_csv(str(outpath),index=False,header=False)

def unit_test():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--client_num", type=int, default=10, required=False)#
    parser.add_argument("-cd", "--check_date", type=str, default="", required=False)#
    #parser.add_argument("-d", "--device", type=str, default="cpu", required=False)
    parser.add_argument("-ml", "--model_name", type=str, default="testmodel.pth", required=False)
    parser.add_argument("-se", "--server_epoch", type=int, default=10, required=False)#
    parser.add_argument("-ce", "--client_epoch", type=int, default=10, required=False)#
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.01, required=False)#
    parser.add_argument("-op", "--optim", type=str, default="SGD", required=False)
    parser.add_argument("-ls", "--loss_func", type=str, default="cross", required=False)
    parser.add_argument("-re", "--reserve_part", type=float, default=0.8, required=False)#
    parser.add_argument("-bc", "--batch_size", type=int, default=10, required=False)#
    parser.add_argument("-tb", "--test_size", type=int, default=100, required=False)
    parser.add_argument("-dt", "--data", type=str, default="../../data/", required=False)#

    args = parser.parse_args()
    arg = arguments(args)
    data = vars(arg)
    del data["path"]
    del data["default"]
    
    #print(data)
    arg.save_backup("./test.txt",data)

if __name__ == "__main__":
    unit_test()