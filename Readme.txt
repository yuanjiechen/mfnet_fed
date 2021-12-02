
The folder contains the source code and environment used in the experiments of our paper. Please download the dataset from: https://drive.google.com/file/d/1p6s_L3QyVf3xW2rCf1Wj-wjnyWsAwvLz/view?usp=sharing

1. Hardware and software requirements
    a. This code requires at least 1 NVidia GPU with VRAM >= 8 GB
    b. We have only tested the code on ubuntu 20.04, Cuda 11.2, and conda 4.9.2
    c. Under the default settings, all the experiments take 3.5 hours in total to complete (we use an E5 server with 4 NVidia 1080Ti GPUs)
    d. We employ TCP protocol for communications. The default TCP port is 8888. If there is any port conflict, please specify the port number with arguments like: -p 9999 

2. Directory structure:
    Please create a folder, say exp/. Please uncompress both the data.zip and hpfl_sup.zip under exp. This leads to the following directory structure. 

./exp
    ./data                  Dataset folder
        distribution        Both i.i.d. and non-i.i.d. sample distributions for training (8 clients)
        images              All images
        labels              One channel labels (value range: 0-8)
        visual              Visualized images
        train.txt           Training set image filenames with non-i.i.d sample distribution (default)
        train-iid.txt       Training set image filenames with i.i.d. sample distribution
        val.txt             Testing set image filenames

    ./hpfl                  Code folder
        experiment.yaml     Experiment conda environment
        start_training.py   Entry point for starting all experiments
        core                Server and client main functions
        model               Server and client model structures
        dataprocess         Dataset initialization
        util                Parameters and connection establishment utilities
        log                 Log files
        cache               Cache data

3. Environment setup
    a: Please install the GPU driver, anaconda, and cuda
    b. Please set up the environment with commands 
        cd ./exp/hpfl
        conda env create -f environment.yaml
        conda activate exp_env
    c. Augment lines 275-278 of start_training.py to specify which GPU to use for training (assuming you have multiple GPUs on your server)

4. Start the experiments
    Please find sample commands used in our evaluations below:

    a. Base experiments
        1. Centralized learning
        python start_training.py -c 1 -re 1.0 -t centralized.csv

        2. FedAvg with default settings (iid)
        python start_training.py -c 8 -re 1.0 -iid 1 -t fedavg_iid.csv

        3. FedAvg with default settings (non-iid)
        python start_training.py -c 8 -re 1.0 -t fedavg_noniid.csv

        4. HPFL with default settings (iid)
        python start_training.py -c 8 -iid 1 -t 8_0.5_ED_0.1_iid.csv

        5. HPFL with default settings (non-iid)
        python start_training.py -c 8 -t 8_0.5_ED_0.1_noniid.csv

        6. HPFL with different alpha values, lambda values, and distillation methods
        python start_training.py -c 8 -re 0.3 -l 0.2 -d 1 -t 8_0.3_DD_0.2_noniid.csv 
        -re specifies the alpha value
        -l specifies the lambda value
        -d specifies the distillation method

    b. Advanced FL algorithms experiments
        1. FedProx with default settings (non-iid)
        python start_training.py -c 8 -prox 0.001 -ce 10 -t prox0.001_noniid.csv
        -prox specifies the mu value of FedProx (where 0 means disabling FedProx)

        2. FedAdam with default settings (non-iid)
        python start_training.py -c 8 -adam 1 -rd 250 -t adam_noniid.csv

        3. FedProx+FedAvg
        python start_training.py -c 8 -re 1.0 -ce 10 -prox 0.001 -t fedavg_prox0.001_noniid.csv

        4. FedAdam+FedAvg
        python start_training.py -c 8 -re 1.0 -rd 250 -adam 1 -t fedavg_adam_noniid.csv
    
        5. Fedprox+HPFL
        python start_training.py -c 8 -re 0.5 -ce 10 -prox 0.001 -t 0.5_prox0.001_noniid.csv

        6. FedAdam+HPFL
        python start_training.py -c 8 -re 0.3 -rd 250 -adam 1 -t 0.3_adam_noniid.csv

5. Default arguments
    We set the default arguments as follows:
        number of client: 8
        server epoch: 3
        client epoch: 3
        total round: 30
        initial learning rate: 0.01 
        decay: round ^2 * 0.95
        alpha value: 0.5 (represents reserve_part in argument parser; set it to 1.0 to fallback to FedAvg)
        dataset path: ../data/
        global batch size: 6
        lambda value: 0.1 (represents distill_param in argument parser)
        tag (result filename): result.csv
        distillation method: ED
        sample distribution: non-i.i.d.
        GPU: cuda:0
        Adam: off
        Prox: off

    These arguments can be changes. Please use "python start_training.py -h" for more details.

6. Output csv format
    We provide the following columns in the output csv files. Their names are self-explanatory.
        effective_accuracy
        accuracy
        test loss
        class0~class9 accuracy
        class1~class9 IoU
        class0~class9 precision
        inference time

