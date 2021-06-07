import logging
from pathlib import Path
import time
import shutil
import argparse
import sys
sys.path.append("..")

from util.parameter import arguments
from util.getlog import get_log

logger = get_log()
logger.setLevel(logging.INFO)
class checker():
    def __init__(self, args) -> None:
        self.args = args
        self.home = Path("./cache")
        self.check_point = self.args.check_point
        self.check_date = self.args.check_date

        self.path = {}

        if not self.home.exists():
            self.home.mkdir(exist_ok=True)
        else:
            pass



    def server_path(self):
        times = 1
        server_home = self.home.joinpath("server")
        server_home.mkdir(exist_ok=True)

        date = str(time.strftime("%Y-%m-%d", time.localtime()))
        ph = date + f"-{times}"
        temppath = server_home.joinpath(ph)
        
        while temppath.exists():
            times += 1
            ph = date + f"-{times}"
            temppath = server_home.joinpath(ph)
        
        if self.check_point == False:
            logger.info(f"New training saved {temppath}")
            temppath.mkdir(exist_ok=False)
        elif self.check_point == True:
            
            if self.check_date == "":
                times -= 1
                ph = date + f"-{times}"
                temppath = server_home.joinpath(ph)

            else:
                temppath = server_home.joinpath(self.check_date)
                logger.info(f"Recover training saved {temppath}")
                print(temppath)
                if not temppath.exists():
                    print("check point fail")
                    return False


        temppath.joinpath("train").mkdir(exist_ok=True)
        temppath.joinpath("weight").mkdir(exist_ok=True)
        temppath.joinpath("log").mkdir(exist_ok=True)
        temppath.joinpath("result").mkdir(exist_ok=True)

        self.path["train"] = temppath.joinpath("train")
        self.path["weight"] = temppath.joinpath("weight")
        self.path["log"] = temppath.joinpath("log")
        self.path["result"] = temppath.joinpath("result")

        if self.path["result"].joinpath("hyperparameter.txt").exists():
            arg = arguments(
                args=self.args,
                path=self.path["result"].joinpath("hyperparameter.txt")
            )
            self.args = arg
        else :
            arg = self.args


        return arg, self.path


    def client_path(self):
        client_num = self.args.client_num
        client_home = self.home.joinpath("client")
        if client_home.exists():
            shutil.rmtree(client_home)
        client_home.mkdir(exist_ok=True)
        for i in range(client_num):
            now_client = client_home.joinpath(f"client-{i}")
            if now_client.exists():
                shutil.rmtree(now_client)
            now_client.mkdir(exist_ok=False)
            self.path[str(i)] = now_client

        return
        
def unit_test():
    ph = Path(__file__).absolute().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--client_num", type=int, default=10, required=False)
    #parser.add_argument("-ca", "--cache_path", type=str, default="../cache", required=False)
    parser.add_argument("-ck", "--check_point", type=bool, default=False, required=False)
    parser.add_argument("-cd", "--check_date", type=str, default="", required=False)
    args = parser.parse_args()
    ps = checker(args=args)
    ps.server_path()
    print(ps.path)

    # cs = checker(args=args)
    # cs.client_path()
    # print(cs.path)


if __name__ == "__main__":
    unit_test()