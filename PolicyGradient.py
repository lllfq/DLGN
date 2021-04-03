# -*- coding:utf-8 -*-
import os
import sys
import random
import math
import tqdm
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from rdkit import  RDLogger

from dataset import GeneratorData,DiscriminatorData
from model.generator import Generator
from model.discriminator import Discriminator
from model.reinforcement import Rollout,PolicyGradient
from utils import read_smiles_from_file,GANLoss,get_reward,canonical_smiles

from fcd import get_fcd, load_ref_model,get_predictions, calculate_frechet_distance


# ================== Parameter Definition =================

parser = argparse.ArgumentParser(description='Training Parameter')
parser.add_argument('--cuda', action='store', default = 0, type=int)
parser.add_argument('--use_cuda', action='store', default = None, type=int)

parser.add_argument('--g_layers',action='store',default = 2, type=int)
parser.add_argument('--g_embed_size',action='store',default = 300, type=int)
parser.add_argument('--g_hidden_size',action='store',default = 1024, type=int)
parser.add_argument('--g_lr',action='store',default = 0.000001, type=float)
parser.add_argument('--g_temperature',action='store',default = 1, type=float)
parser.add_argument('--g_temperature_decay',action='store',default = 0.995, type=float)

parser.add_argument('--d_embed_size',action='store',default = 100, type=int)
parser.add_argument('--d_hidden_size',action='store',default = 300, type=int)
parser.add_argument('--d_lr',action='store',default = 0.00001, type=float)
parser.add_argument('--d_dropout',action='store',default = 0.2, type=float)
parser.add_argument('--d_batch_size', action='store', default = 64, type=int)
parser.add_argument('--d_pre_epoch',action='store',default = 15, type=int)

parser.add_argument('--epochs_num',action='store',default = 50, type=int)
parser.add_argument('--g_epoch',action='store',default = 1000, type=int)
parser.add_argument('--d_epoch',action='store',default = 20, type=int)

parser.add_argument('--print_every',action='store',default = 100 , type=int)
parser.add_argument('--beta',action='store',default=0.3,type=int)

parser.add_argument('--model_path', action='store', default = 'model_saved/MLE', type=str)
parser.add_argument('--save_path', action='store', default = 'model_saved/PolicyGradient/Psy_50_1000_20_5_4_0.3', type=str)
parser.add_argument('--g_name', action='store', default = 'MLE_generator.pt', type=str)
parser.add_argument('--g_data_file', action='store', default = 'Chembl24.txt', type=str)
parser.add_argument('--train_file_1', action='store', default = 'DRD2_train.txt', type=str)
parser.add_argument('--train_file_2', action='store', default = 'HTR1A_train.txt', type=str)
parser.add_argument('--valid_file_1', action='store', default = 'DRD2_valid.txt', type=str)
parser.add_argument('--valid_file_2', action='store', default = 'HTR1A_valid.txt', type=str)

opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# =========================================================
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
if opt.use_cuda is None:
    opt.use_cuda  = torch.cuda.is_available()    

if opt.cuda is not None and opt.cuda >= 0:
    torch.cuda.set_device(opt.cuda)

tokens = ['?','<','>','#', '(', ')', '+', '-', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', 'B', 'C', 'F', 'H', 'I', 'N', 'O', 'S', '[', ']', 'c', 'l', 'n', 'o', 'r', 's']
char_num = len(tokens)

optimizer_instance = torch.optim.Adam

if __name__ == '__main__':
    # Define Networks
    print('---------- Defining and loading Network')
    generator = Generator(input_size = char_num,embed_size = opt.g_embed_size, hidden_size = opt.g_hidden_size,
                    output_size = char_num, n_layers=opt.g_layers,use_cuda=opt.use_cuda, 
                    optimizer_instance=optimizer_instance, lr=opt.g_lr)
    discriminator_1 = Discriminator(input_size = char_num,embed_size = opt.d_embed_size, hidden_size = opt.d_hidden_size,
                                use_cuda = opt.use_cuda, dropout = opt.d_dropout, lr = opt.d_lr ,optimizer_instance = torch.optim.Adam)
    discriminator_2 = Discriminator(input_size = char_num,embed_size = opt.d_embed_size, hidden_size = opt.d_hidden_size,
                                use_cuda = opt.use_cuda, dropout = opt.d_dropout, lr = opt.d_lr ,optimizer_instance = torch.optim.Adam)
    fcd_model = load_ref_model()
    # Load pretrained generator 
    generator.load_model(os.path.join(os.getcwd(),opt.model_path,opt.g_name),map_location='cuda:0')
    
    # Define path
    if not os.path.exists(os.path.join(os.getcwd(),opt.save_path)):
        os.makedirs(os.path.join(os.getcwd(),opt.save_path))
    reward_d1_path = os.path.join(os.getcwd(),opt.save_path,'A_reward_d1.txt')
    reward_d2_path = os.path.join(os.getcwd(),opt.save_path,'A_reward_d2.txt')
    reward_valid_path = os.path.join(os.getcwd(),opt.save_path,'A_reward_valid.txt')
    nll_1_path = os.path.join(os.getcwd(),opt.save_path,'A_nll_1.txt')
    nll_2_path = os.path.join(os.getcwd(),opt.save_path,'A_nll_2.txt')
    fcd_1_path = os.path.join(os.getcwd(),opt.save_path,'A_fcd_1.txt')
    fcd_2_path = os.path.join(os.getcwd(),opt.save_path,'A_fcd_2.txt')
    model_d1_path = os.path.join(os.getcwd(),opt.save_path,'A_discriminator_1.pt')
    model_d2_path = os.path.join(os.getcwd(),opt.save_path,'A_discriminator_2.pt')
    model_g_path = os.path.join(os.getcwd(),opt.save_path,'A_generator.pt')
    log_path_d1 = os.path.join(os.getcwd(),opt.save_path,'A_dis_1_loss.txt')
    log_path_d2 = os.path.join(os.getcwd(),opt.save_path,'A_dis_2_loss.txt')

    # Define GeneratorData
    print('---------- Loading GeneratorData')
    gen_loader = GeneratorData(os.path.join(os.getcwd(),'data',opt.g_data_file),tokens = tokens,use_cuda = opt.use_cuda)
    eval_loader_1 =  GeneratorData(os.path.join(os.getcwd(),'data',opt.valid_file_1),tokens = tokens,use_cuda = opt.use_cuda)
    eval_loader_2 =  GeneratorData(os.path.join(os.getcwd(),'data',opt.valid_file_2),tokens = tokens,use_cuda = opt.use_cuda)

    
    # Read truth data for discriminator
    truth_data_path_1 = os.path.join(os.getcwd(),'data',opt.train_file_1)
    truth_data_path_2 = os.path.join(os.getcwd(),'data',opt.train_file_2)
    truth_data_1,_ = read_smiles_from_file(truth_data_path_1)
    truth_data_2,_ = read_smiles_from_file(truth_data_path_2)

    # Use Generator to generate some fake data for discriminator pretraining
    fake_data = []
    num = 0 
    fake_data_len = 10000
    print('---------- Using generator to generate fake data for discriminiator pretraining')
    with torch.no_grad():
        while(num < fake_data_len):
            sample = generator.generate(gen_loader)
            if len(sample)==2:
                continue
            else:
                if sample[-1]=='>':
                    fake_data.append(sample[1:-1])
                else :
                    fake_data.append(sample[1:])
                num = num + 1
    
    # evaluate pretrained generator
    print('---------- Evaluating pretrained generator')
    real_sample_1 = [s[1:-1] for s in eval_loader_1.smiles_list if canonical_smiles(s[1:-1]) is not None]
    real_sample_2 = [s[1:-1] for s in eval_loader_2.smiles_list if canonical_smiles(s[1:-1]) is not None]
    act_real_1 = get_predictions(fcd_model, real_sample_1)
    act_real_2 = get_predictions(fcd_model, real_sample_2)
    mu_real_1 = np.mean(act_real_1, axis=0)
    sigma_real_1 = np.cov(act_real_1.T)
    mu_real_2 = np.mean(act_real_2, axis=0)
    sigma_real_2 = np.cov(act_real_2.T)

    fake_sample = [w for w in fake_data if canonical_smiles(w) is not None]
    act_fake = get_predictions(fcd_model, fake_sample)
    mu_fake = np.mean(act_fake, axis=0)
    sigma_fake = np.cov(act_fake.T)

    fcd_score_1 = calculate_frechet_distance(
        mu1=mu_real_1,
        mu2=mu_fake, 
        sigma1=sigma_real_1,
        sigma2=sigma_fake)

    fcd_score_2 = calculate_frechet_distance(
        mu1=mu_real_2,
        mu2=mu_fake, 
        sigma1=sigma_real_2,
        sigma2=sigma_fake)
    f = open(fcd_1_path,'a')
    f.write(str(fcd_score_1)+"\n")
    f.close()
    f = open(fcd_2_path,'a')
    f.write(str(fcd_score_2)+"\n")
    f.close()
    print('FCD_1: ',fcd_score_1)
    print('FCD_2: ',fcd_score_2)
    with torch.no_grad():
        mean_nll_1 = generator.evaluate(eval_loader = eval_loader_1,log_path = nll_1_path)
        mean_nll_2 = generator.evaluate(eval_loader = eval_loader_2,log_path = nll_2_path)
    print('Mean negative log likelihood on eval_dataset_1: {}'.format(mean_nll_1))
    print('Mean negative log likelihood on eval_dataset_2: {}'.format(mean_nll_2))
    min_fcd_1 = fcd_score_1
    min_fcd_2 = fcd_score_2
    # Define DiscriminatorData
    print('---------- Loading DiscriminatorData')
    random.shuffle(fake_data)
    dis_loader1 = DiscriminatorData(truth_data=truth_data_1,fake_data=fake_data[0:len(truth_data_1)],tokens=tokens,batch_size=opt.d_batch_size)
    random.shuffle(fake_data)
    dis_loader2 = DiscriminatorData(truth_data=truth_data_2,fake_data=fake_data[0:len(truth_data_2)],tokens=tokens,batch_size=opt.d_batch_size)
    # Pretrain Discriminator
    print('---------- Pretrain Discriminator ...')
    discriminator_1.train()
    discriminator_2.train()
    loss_1 = discriminator_1.train_epochs(dis_loader1,opt.d_pre_epoch)
    loss_2 = discriminator_2.train_epochs(dis_loader2,opt.d_pre_epoch)
    f = open(log_path_d1,'a')
    for l in loss_1:
        f.write(str(l)+"\n")
    f.close()
    f = open(log_path_d2,'a')
    for l in loss_2:
        f.write(str(l)+"\n")
    f.close()
    
    # Adversarial Training
    policy = PolicyGradient(gen_loader,beta=opt.beta)
    if opt.use_cuda:
        ganloss = GANLoss().cuda()
    else :
        ganloss = GANLoss()
    print('---------- Start Adeversatial Training...')
    for epoch in range(opt.epochs_num):
        # Train the generator for g
        discriminator_1.eval()
        discriminator_2.eval()
        print('---------- Training generator')
        total_d1 = 0
        total_d2 = 0
        total_valid = 0
        generator.optimizer.zero_grad()
        for i in range(opt.g_epoch):
            # generate a sample
            with torch.no_grad():
                sample = generator.generate(gen_loader)
                # calculate the reward
                reward = policy.get_reward(x=sample,discriminator1=discriminator_1,discriminator2=discriminator_2,use_cuda=opt.use_cuda)
                reward_d1,reward_d2,reward_valid = get_reward(sample,discriminator_1,discriminator_2,gen_loader)
                total_d1 = total_d1 + reward_d1
                total_d2 = total_d2 + reward_d2
                total_valid = total_valid + reward_valid
            # calculate the loss and optimize
            prob = generator.get_prob(gen_loader.char_tensor(sample))
            g_loss = ganloss(prob,reward)
            g_loss.backward() 
            generator.optimizer.step()
            generator.optimizer.zero_grad()
            if (i+1) % opt.print_every == 0 and i != 0 :
                print('Adversatial epoch:{}/{}, Generator epoch:{}/{}'.format(epoch+1,opt.epochs_num,i+1,opt.g_epoch))
                print('Reward:{}  {}  {}'.format(total_d1/opt.print_every,total_d2/opt.print_every,total_valid/opt.print_every))
                            # save reward to log file
                f = open(reward_d1_path,'a')
                f.write(str(total_d1/opt.print_every)+"\n")
                f.close()
                f = open(reward_d2_path,'a')
                f.write(str(total_d2/opt.print_every)+"\n")
                f.close()
                f = open(reward_valid_path,'a')
                f.write(str(total_valid/opt.print_every)+"\n")
                f.close()

                total_d1 = 0
                total_d2 = 0
                total_valid = 0
        

        
        #evaluate generator for nll
        print('---------- Evaluating generator with nll ...')
        with torch.no_grad():
            mean_nll_1 = generator.evaluate(eval_loader = eval_loader_1,log_path = nll_1_path)
            mean_nll_2 = generator.evaluate(eval_loader = eval_loader_2,log_path = nll_2_path)
        print('Mean negative log likelihood on eval_dataset_1: {}'.format(mean_nll_1))
        print('Mean negative log likelihood on eval_dataset_2: {}'.format(mean_nll_2))


        # Train the discriminator 
        # generate fake data for discriminator
        print('---------- Generating fake data for discriminator ...')
        fake_data = []
        num = 0 
        with torch.no_grad():
            while(num < fake_data_len):
                sample = generator.generate(gen_loader)
                if len(sample)==2:
                    continue
                else:
                    if sample[-1]=='>':
                        fake_data.append(sample[1:-1])
                    else :
                        fake_data.append(sample[1:])
                    num = num + 1

        #evaluate generator for fcd
        print('---------- Calculating fc distance between generating and real data ...' )
        fake_sample = [w for w in fake_data if canonical_smiles(w) is not None]
        act_fake = get_predictions(fcd_model, fake_sample)
        mu_fake = np.mean(act_fake, axis=0)
        sigma_fake = np.cov(act_fake.T)

        fcd_score_1 = calculate_frechet_distance(
            mu1=mu_real_1,
            mu2=mu_fake, 
            sigma1=sigma_real_1,
            sigma2=sigma_fake)

        fcd_score_2 = calculate_frechet_distance(
            mu1=mu_real_2,
            mu2=mu_fake, 
            sigma1=sigma_real_2,
            sigma2=sigma_fake)
        f = open(fcd_1_path,'a')
        f.write(str(fcd_score_1)+"\n")
        f.close()
        f = open(fcd_2_path,'a')
        f.write(str(fcd_score_2)+"\n")
        f.close()
        print('FCD_1: ',fcd_score_1)
        print('FCD_2: ',fcd_score_2)
        print('Min FCD_1: {}'.format(min_fcd_1))
        print('Min FCD_2: {}'.format(min_fcd_2))
        if fcd_score_1 < min_fcd_1 and fcd_score_2 < min_fcd_2 :
            min_fcd_1 = fcd_score_1
            min_fcd_2 = fcd_score_2
            print("Model saved")
            generator.save_model(model_g_path)

        print('---------- Train Discriminator ...')
        random.shuffle(fake_data)
        dis_loader1.update(truth_data = None, fake_data = fake_data[0:len(truth_data_1)])
        random.shuffle(fake_data)
        dis_loader2.update(truth_data = None, fake_data = fake_data[0:len(truth_data_2)])
        discriminator_1.train()
        discriminator_2.train()
        loss_1 = discriminator_1.train_epochs(dis_loader1,opt.d_epoch)
        loss_2 = discriminator_2.train_epochs(dis_loader2,opt.d_epoch)
        f = open(log_path_d1,'a')
        for l in loss_1:
            f.write(str(l)+"\n")
        f.close()
        f = open(log_path_d2,'a')
        for l in loss_2:
            f.write(str(l)+"\n")
        f.close()
        discriminator_1.save_model(model_d1_path)
        discriminator_2.save_model(model_d2_path)