import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import NLLLoss,canonical_smiles
from fcd import get_fcd, load_ref_model,get_predictions, calculate_frechet_distance

from tqdm import trange

class Generator(nn.Module):
    def __init__(self, input_size, embed_size , hidden_size, output_size,
                n_layers=2,use_cuda=None,optimizer_instance=torch.optim.Adadelta, lr=0.01):
        """
        Parameters
        ----------
        input_size: int
            number of characters in the alphabet

        embed_size: int
            size of word embedding

        hidden_size: int
            size of the RNN layer(s)

        output_size: int
            again number of characters in the alphabet

        n_layers: int (default 1)
            number of RNN layers

        use_cuda: bool (default None)
            parameter specifying if GPU is used for computations. If left
            unspecified, GPU will be used if available

        optimizer_instance: torch.optim object (default torch.optim.Adadelta)
            optimizer to be used for training

        lr: float (default 0.01)
            learning rate for the optimizer

        """
        super(Generator, self).__init__()
        
        self.input_size = input_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        
        self.encoder = nn.Embedding(input_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, n_layers)
        self.decoder = nn.Sequential(nn.Linear( hidden_size , int(hidden_size/2) ), nn.LeakyReLU(0.1),nn.Linear(int(hidden_size/2),output_size))

        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        
        self.use_cuda = use_cuda
        if self.use_cuda is None:
            self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self = self.cuda()
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.optimizer_instance = optimizer_instance
        self.optimizer = self.optimizer_instance(self.parameters(), lr=lr,
                                                 weight_decay=0.00001)
  
    def load_model(self, path, map_location = None):
        """
        Loads pretrained parameters from the checkpoint into the model.

        Parameters
        ----------
        path: str
            path to the checkpoint file model will be loaded from.
        """
        if map_location is None:
            weights = torch.load(path)
        else :
            weights = torch.load(path, map_location=map_location)
            
        self.load_state_dict(weights)

    def save_model(self, path):
        """
        Saves model parameters into the checkpoint file.

        Parameters
        ----------
        path: str
            path to the checkpoint file model will be saved to.
        """
        torch.save(self.state_dict(), path)

    def change_lr(self, new_lr):
        """
        Updates learning rate of the optimizer.

        Parameters
        ----------
        new_lr: float
            new learning rate value
        """
        self.optimizer = self.optimizer_instance(self.parameters(), lr=new_lr)
        self.lr = new_lr

    def init_hidden(self):
        """
        Initialization of the hidden state of RNN.

        Returns
        -------
        hidden: torch.tensor
            tensor filled with zeros of an appropriate size (taking into
            account number of RNN layers )
        """
        if self.use_cuda:
            return torch.zeros(self.n_layers, 1, self.hidden_size).cuda()
        else:
            return torch.zeros(self.n_layers, 1, self.hidden_size)

    def init_cell(self):
        """
        Initialization of the cell state of LSTM. 

        Returns
        -------
        cell: torch.tensor
            tensor filled with zeros of an appropriate size (taking into
            account number of RNN layers)
        """
        if self.use_cuda:
            return torch.zeros(self.n_layers, 1, self.hidden_size).cuda()
        else:
            return torch.zeros(self.n_layers, 1, self.hidden_size)

    def forward(self, inp, hidden):
        """
        Forward one step of the model. Generates probability of the next character
        given the prefix.

        Parameters
        ----------
        inp: torch.tensor
            input tensor that contains prefix string indices

        hidden: tuple(torch.tensor, torch.tensor)
            previous hidden state of the model. hidden is a tuple of hidden state and cell state

        Returns
        -------
        output: torch.tensor
            tensor with non-normalized probabilities of the next character

        next_hidden:  tuple(torch.tensor, torch.tensor)
            next hidden state of the model. next_hidden is a tuple of hidden state and cell state

        """
        inp = self.encoder(inp.view(1, -1))
        output, next_hidden = self.rnn(inp.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, next_hidden

    def generate(self, dataloader, prime_str='<', end_token='>', predict_len=150,temperature=1):
        """
        Generates new string from the model distribution.

        Parameters
        ----------
        dataloader: object of type GeneratorData
            stores information about the generator dataloader format such alphabet, etc

        prime_str: str (default '<')
            prime string that will be used as prefix. Default value is just the
            START_TOKEN

        end_token: str (default '>')
            when end_token is sampled from the model distribution,
            the generation of a new example is finished

        predict_len: int (default 150)
            maximum length of the string to be generated. If the end_token is
            not sampled, the generation will be aborted when the length of the
            generated sequence is equal to predict_len

        Returns
        -------
        new_sample: str
            Newly generated sample from the model distribution.

        """
        hidden = self.init_hidden()
        cell = self.init_cell()
        hidden = (hidden, cell)

        prime_input = dataloader.char_tensor(prime_str)
        new_sample = prime_str

        # Use priming string to "build up" hidden state
        for p in range(len(prime_str)-1):
            _, hidden = self.forward(prime_input[p], hidden)
        inp = prime_input[-1]

        for p in range(predict_len-len(prime_str)):
            output, hidden = self.forward(inp, hidden)

            # Sample from the network as a multinomial distribution
            probs = torch.softmax(output/temperature, dim=1)
            sampled_index = torch.multinomial(probs.view(-1), 1)[0].cpu().numpy()

            # Add predicted character to string and use as next input
            predicted_char = dataloader.all_characters[sampled_index]
            new_sample += predicted_char
            inp = dataloader.char_tensor(predicted_char)
            if predicted_char == end_token:
                break
        return new_sample

    def train_one_smiles(self, inp, target):
        """
        One train step, i.e. forward-backward and parameters update, for
        a single training example.

        Parameters
        ----------
        inp: torch.tensor
            tokenized training string from position 0 to position (seq_len - 1)

        target:
            tokenized training string from position 1 to position seq_len

        Returns
        -------
        loss: float
            mean value of the loss function (averaged through the sequence
            length)

        """
        hidden = self.init_hidden()
        cell = self.init_cell()
        hidden = (hidden, cell)

        self.optimizer.zero_grad()
        loss = 0
        for c in range(len(inp)):
            output, hidden = self(inp[c], hidden)
            loss += self.criterion(output, target[c].unsqueeze(0))

        loss.backward()
        self.optimizer.step()

        return loss.item() / len(inp)
     
    def pretrain(self, dataloader,epochs,save_path,eval_loader1=None,eval_loader2=None):
        """
        This methods fits the parameters of the model. Training is performed to
        minimize the cross-entropy loss when predicting the next character
        given the prefix.

        Parameters
        ----------
        dataloader: object of type GeneratorData
            stores information about the generator data format such alphabet, etc

        epochs: int
            how many iterations of training will be performed

        save_path:str
            path to save the trained model and loss log

        Returns
        -------
        all_losses: list
            list that stores the values of the loss function (learning curve)
        """
        model_path = os.path.join(save_path,'MLE_generator.pt')
        log_path = os.path.join(save_path,'MLE_loss.txt')
        total_num = epochs * len(dataloader)
        epoch_loss = 0
        num = 0
        print_every = 10000

        ##############################
        fcd_model = load_ref_model()
        if eval_loader1:
            real_sample_1 = [s[1:-1] for s in eval_loader1.smiles_list if canonical_smiles(s[1:-1]) is not None]
            act_real_1 = get_predictions(fcd_model, real_sample_1)
            mu_real_1 = np.mean(act_real_1, axis=0)
            sigma_real_1 = np.cov(act_real_1.T)

        if eval_loader2:
            real_sample_2 = [s[1:-1] for s in eval_loader2.smiles_list if canonical_smiles(s[1:-1]) is not None]
            act_real_2 = get_predictions(fcd_model, real_sample_2)
            mu_real_2 = np.mean(act_real_2, axis=0)
            sigma_real_2 = np.cov(act_real_2.T)
        
        #############################

        print('-------------------------generator pretraining begin')
        for n in trange(1, total_num + 1, desc='Training in progress...',ncols=100):
            (inp, target) = dataloader.next()
            loss = self.train_one_smiles(inp, target)
            epoch_loss += loss
            num += 1
            if num % print_every == 0:
                print('\n-------------------------Epoch loss: [%d/%d (%d%%) %.4f]' % (n,total_num,  num / total_num * 100, epoch_loss/print_every))
                print('-------------------------Model saved at '+ model_path)
                self.save_model(model_path)
                f = open(log_path,'a')
                f.write(str(epoch_loss/print_every)+"\n")
                f.close()
                epoch_loss = 0

                # Use Generator to generate some fake data 
                fake_data = []
                num = 0 
                fake_data_len = 3000
                print('---------- Using generator to generate fake data for discriminiator pretraining')
                with torch.no_grad():
                    while(num < fake_data_len):
                        sample = self.generate(dataloader)
                        if len(sample)==2:
                            continue
                        else:
                            if sample[-1]=='>':
                                fake_data.append(sample[1:-1])
                            else :
                                fake_data.append(sample[1:])
                            num = num + 1
                fake_sample = [w for w in fake_data if canonical_smiles(w) is not None]
                act_fake = get_predictions(fcd_model, fake_sample)
                mu_fake = np.mean(act_fake, axis=0)
                sigma_fake = np.cov(act_fake.T)

                with torch.no_grad():
                    if eval_loader1:
                        fcd_score_1 = calculate_frechet_distance(
                            mu1=mu_real_1,
                            mu2=mu_fake, 
                            sigma1=sigma_real_1,
                            sigma2=sigma_fake)
                        nll_1 = self.evaluate(eval_loader1,os.path.join(save_path,'MLE_nll_1.txt'))
                        f = open(os.path.join(save_path,'MLE_fcd_1.txt'),'a')
                        f.write(str(fcd_score_1)+"\n")
                        f.close()
                        print('-------------------------Mean negative log likelihood of eval_loader_1: '+str(nll_1))
                        print('-------------------------FCD of eval_loader_1: '+str(fcd_score_1))
                    if eval_loader2:
                        fcd_score_2 = calculate_frechet_distance(
                            mu1=mu_real_2,
                            mu2=mu_fake, 
                            sigma1=sigma_real_2,
                            sigma2=sigma_fake)
                        nll_2 = self.evaluate(eval_loader2,os.path.join(save_path,'MLE_nll_2.txt'))
                        f = open(os.path.join(save_path,'MLE_fcd_2.txt'),'a')
                        f.write(str(fcd_score_2)+"\n")
                        f.close()
                        print('-------------------------Mean negative log likelihood of eval_loader_2: '+str(nll_2))
                        print('-------------------------FCD of eval_loader_2: '+str(fcd_score_2))
        print('-------------------------generator pretraining finished')

    def get_prob(self,trajectory):
        """
        follow a input trajectory and computed the log likelihood of each generated char

        Parameters
        ----------
        trajectory : tensor
        a input trajectory has been transformed by char_tensor(trajectory)

        Returns
        -------
        prob:tensor 
            tensor of prob with length of len(trajectory)-1
        """
        inp = trajectory[:-1]
        target = trajectory[1:]
        length = len(inp)
        hidden = self.init_hidden()
        cell = self.init_cell()
        hidden = (hidden, cell)

        inp = self.encoder(inp.view(1, -1))
        out, _ = self.rnn(inp.view(length, 1, -1), hidden)
        out = self.decoder(torch.squeeze(out,1))
        prob_matrix = self.log_softmax(out) # length x output_size
        index = torch.LongTensor(target.view(1,-1).cpu()).t()
        if self.use_cuda:
            index = index.cuda()
        prob = prob_matrix.gather(1,index).view(length)
        return prob

    def evaluate(self,eval_loader,log_path):
        """
        This methods evaluate the model. Training is performed to
        minimize the cross-entropy loss when predicting the next character
        given the prefix.

        Parameters
        ----------
        eval_loader: object of type GeneratorData
            stores information about the generator data format such alphabet, etc

        log_path:str
            path to save the nll 
        Returns
        -------
        mean_nll:int
            
        """
        nll = NLLLoss()
        total_loss = 0
        for trajectory in eval_loader.smiles_list:
            prob = self.get_prob(eval_loader.char_tensor(trajectory))
            loss = nll(prob).item()
            total_loss += loss

        mean_nll = total_loss / eval_loader.smiles_num
        if log_path is not None:
            f = open(log_path,'a')
            f.write(str(mean_nll)+"\n")
            f.close()
        return mean_nll