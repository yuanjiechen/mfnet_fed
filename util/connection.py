import logging
import socket
from threading import Thread
import struct
import os
import argparse
from pathlib import Path

#not complete:
#messager,  logger, status checker
logger = logging.getLogger(__name__)
class connection():
    def __init__(self, args ,send_path, recv_path, ip_addr="0.0.0.0", port=8888):
        self.ip = ip_addr
        self.port = port
        self.sk = None
        self.args = args

        self.conn = []
        self.addr = []

        self.send_path = send_path
        self.recv_path = recv_path

        self.model_name = "testmodel.pkl"
    def establish_conn(self):
        self.sk = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        # while True:
        #     if self.check_port(self.port) == False :
        #         self.port = self.port + 1
        #     else:
        #         break
        try:
            self.sk.bind(('0.0.0.0',self.port))
            self.sk.listen(10)        

            print("Initial success",", port : ",self.port)
        except socket.error as msg:
            print(msg,"Some error,try again")
            raise

        th = Thread(target=self.listener)
        th.setDaemon(True)
        th.start()

    def check_port(self,port):
        sk = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        try:
            sk.connect(('0.0.0.0',port))
            sk.close()
            return False
        except:
            return True

    def listener(self):
        while True:
            conn, addr = self.sk.accept()
            self.conn.append(conn)
            self.addr.append(addr)
            th = Thread(
                target=self.client_comm, 
                args=(conn,)
            )
            th.setDaemon(True)
            th.start()


    def client_comm(self, client):
        try :
            client.setblocking(True)
            client_num = self.conn.index(client)
            client.sendall((str(client_num)).encode(encoding='utf-8'))
        except socket.error as msg:
            print(msg)
            raise

        while True:
            try:
                data = client.recv(2).decode(
                    encoding='utf-8',
                    errors="ignore"
                )

            except ConnectionResetError as msg:
                print(msg)
                client.close()
                self.clientlist.remove(client)
                return

            if data == "IN":
                pass

            elif data == "SR":
                sender(
                    sk=client, 
                    path=str(self.send_path.joinpath(self.model_name))
                )

            elif data == "RE":
                recver(
                    sk=client, 
                    path=str(self.recv_path.joinpath(f"client{client_num}.pth"))
                )
            
            elif data == "DS":
                recver(
                    sk=client,
                    path=str(self.recv_path.joinpath(f"client{client_num}.npy"))
                )



def connector(ip_addr, port):
    sk = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    try:
        sk.connect((str(ip_addr),port))
        sk.setblocking(True)

        num = int(sk.recv(10).decode(encoding='utf-8'))

        return sk, int(num)
    except Exception as msg:
        print(msg)
        raise


def sender(sk,path):

    try:
        fhead = struct.pack('512sl', os.path.basename(path).encode(encoding="utf-8"), os.stat(path).st_size)
    except IOError as msg:
        print("Model not found!")
        raise
    try:
        sk.sendall(fhead)
        file = open(path,'rb')
        while True:
            data = file.read(1024)
            if not data:
                break
            sk.sendall(data)
        
        file.close()
    except (socket.error, IOError) as msg:
        print(msg)
        raise
    return

def recver(sk,path):

    fileinfo_size = struct.calcsize("512sl")
    try:

        buffer = sk.recv(fileinfo_size,socket.MSG_WAITALL)
        filename,filesize = struct.unpack("512sl",buffer)
        filename = filename.strip(b'\00')
        filename = filename.decode(encoding="utf-8")
        

        file_recv = open(path,"wb")
        recv_size = 0
        while not filesize == recv_size:
            if filesize - recv_size > 1024:
                data = sk.recv(1024,socket.MSG_WAITALL)
                recv_size = recv_size + len(data)
            else :
                data = sk.recv(filesize - recv_size)
                recv_size = filesize
            
            file_recv.write(data)  
        file_recv.close()
        #print("Receive success")

    except Exception as msg:#(socket.error,struct.error,IOError) as msg:
        print(msg)
        print("Recv model error!")
        raise


def unit_test():
    ph = Path.cwd() # shell directory
    #Path(__file__) # file directory include file name 
    test_file = ph.joinpath("send_test.txt")
    recv_file = ph.joinpath("recv.txt")
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--model_path", type=str, default=str(test_file), required=False)
    parser.add_argument("-r", "--recv_path", type=str, default=str(recv_file), required=False)
    args = parser.parse_args()
    cn = connection(args)
    cn.establish_conn()

    client, num = connector("140.114.89.42",8888)
    client.send("SR".encode(encoding="utf-8"))
    recver(client, args.recv_path)
    print(num)
    input()

if __name__ == "__main__":
    unit_test()
