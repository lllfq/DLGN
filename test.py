import os,sys,random,math,warnings,umap,argparse,matplotlib
sys.path.append("..")

import torch
import numpy as np
import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler
from rdkit.Chem import AllChem as Chem
from utils import read_smiles_from_file,canonical_smiles
from dataset import GeneratorData
from model.generator import Generator
import moses

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
matplotlib.use("TKAgg")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ================== Parameter Definition =================
parser = argparse.ArgumentParser(description='Do mose')

parser.add_argument('--cuda', action='store', default = 0, type=int)
parser.add_argument('--use_cuda', action='store', default = None, type=int)
parser.add_argument('--g_layers',action='store',default = 2, type=int)
parser.add_argument('--g_embed_size',action='store',default = 300, type=int)
parser.add_argument('--g_hidden_size',action='store',default = 1024, type=int)
parser.add_argument('--target_file_1', type=str,default='DRD2_test.txt')
parser.add_argument('--target_file_2', type=str,default='HTR1A_test.txt')
parser.add_argument('--train_file', type=str,default='Psy_train.txt')
parser.add_argument('--model_file', type=str,default='model_saved/PolicyGradient/Psy_50_1000_20_5_4_0.3/A_generator.pt')
parser.add_argument('--data_len',type=int,default=30000)

if __name__ == '__main__':
    ####################################
    opt = parser.parse_args()

    if opt.use_cuda is None:
        opt.use_cuda  = torch.cuda.is_available()    

    if opt.cuda is not None and opt.cuda >= 0:
        torch.cuda.set_device(opt.cuda)

    tokens = ['?','<','>','#', '(', ')', '+', '-', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', 'B', 'C', 'F', 'H', 'I', 'N', 'O', 'S', '[', ']', 'c', 'l', 'n', 'o', 'r', 's']
    char_num = len(tokens)

    # read generator 
    print('Loading generator')
    model_path = os.path.join(os.getcwd(),'..',opt.model_file)
    
    model_generator = Generator(input_size = char_num,embed_size = opt.g_embed_size, hidden_size = opt.g_hidden_size,
                    output_size = char_num, n_layers=opt.g_layers,use_cuda=opt.use_cuda)


    model_generator.load_model(path = model_path ,map_location='cuda:0')

    # read source data and target data 
    print('Reading source data and target data')

    target_path_1 = os.path.join(os.getcwd(),'../data',opt.target_file_1)
    target_path_2 = os.path.join(os.getcwd(),'../data',opt.target_file_2)

    target_smi_1,_ = read_smiles_from_file(target_path_1)
    target_smi_2,_ = read_smiles_from_file(target_path_2)

    train_path = os.path.join(os.getcwd(),'../data',opt.train_file)
    train_smi,_ = read_smiles_from_file(train_path)

    gen_loader = GeneratorData(os.path.join(os.getcwd(),'../data',opt.target_file_1),tokens = tokens,use_cuda = opt.use_cuda)
    
    # generate samples
    print('Generating model data')
    
    ############################################################
    model_smi = []
    num = 0
    with torch.no_grad():
        while(num < opt.data_len):
            sample = model_generator.generate(gen_loader,temperature=0.5)
            if len(sample)==2:
                continue
            else:
                if sample[-1]=='>':
                        model_smi.append(sample[1:-1])
                        num = num + 1
                else :
                        model_smi.append(sample[1:])
                        num = num + 1
            if num%1000==0:
                print("Generating {}/30000".format(num))

    print('Caculating model Metric')
    model_metrics = moses.get_all_metrics(gen=model_smi,test=target_smi_1,test_scaffolds=target_smi_2,train = train_smi)
    print(model_metrics)
    
    
    
