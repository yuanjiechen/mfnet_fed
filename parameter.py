import csv
import pandas as pd
import numpy as np

class pm():
    def __init__(self,Pm_file="/home/yuanjie/mf_fed/hyperparameter.txt"):
        try:
            self.csv = pd.read_csv(Pm_file, encoding="utf-8",header=None)
            
        except Exception as msg:
            print(msg)
            raise
    
    def getdata(self):
        data_dict = {}
        origin_data = self.csv.to_dict(orient="records")

        for data in origin_data:
            data_dict[data[0]] = data[1]

        return data_dict