# -*- coding:utf-8 -*-
import os
import random
import math
import tqdm
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model.generator import Generator
from dataset import GeneratorData


# ================== Parameter Definition =================

parser = argparse.ArgumentParser(description='Training Parameter')
parser.add_argument('--cuda', action='store', default = 0, type=int)
parser.add_argument('--use_cuda', action='store', default = None, type=int)
parser.add_argument('--save_path', action='store', default = 'model_saved/MLE', type=str)
parser.add_argument('--layers',action='store',default = 2, type=int)
parser.add_argument('--embed_size',action='store',default = 300, type=int)
parser.add_argument('--hidden_size',action='store',default = 1024, type=int)
parser.add_argument('--lr',action='store',default = 0.00001, type=float)
parser.add_argument('--epoch',action='store',default = 10, type=int)
opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

if opt.cuda is not None and opt.cuda >= 0:
    torch.cuda.set_device(opt.cuda)


if not os.path.exists(os.path.join(os.getcwd(),opt.save_path)):
        os.makedirs(os.path.join(os.getcwd(),opt.save_path))


tokens = ['?','<','>','#', '(', ')', '+', '-', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', 'B', 'C', 'F', 'H', 'I', 'N', 'O', 'S', '[', ']', 'c', 'l', 'n', 'o', 'r', 's']

gen_data_path = os.path.join(os.getcwd(),'data','Chembl24.txt')
eval_data_path1 = os.path.join(os.getcwd(),'data','DRD2_valid.txt')
eval_data_path2 = os.path.join(os.getcwd(),'data','HTR1A_valid.txt')
optimizer_instance = torch.optim.Adam

# Load data from file
gen_loader = GeneratorData(gen_data_path,tokens = tokens)
eval_loader1 = GeneratorData(eval_data_path1,tokens = tokens)
eval_loader2 = GeneratorData(eval_data_path2,tokens = tokens)

# Define network
generator = Generator(input_size = gen_loader.char_num,embed_size = opt.embed_size, hidden_size = opt.hidden_size,
                    output_size = gen_loader.char_num, n_layers=opt.layers,use_cuda=opt.use_cuda, 
                    optimizer_instance=optimizer_instance, lr=opt.lr)

# train Generator using MLE
save_path = os.path.join(os.getcwd(),opt.save_path)
generator.pretrain(dataloader = gen_loader,epochs = opt.epoch, save_path = save_path, 
                    eval_loader1 =  eval_loader1 ,eval_loader2 =  eval_loader2)

