import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from tqdm import trange

class Discriminator(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size,use_cuda=None, dropout=0.2,optimizer_instance=torch.optim.Adadelta, lr=0.001):
        """
        Parameters
        ----------
        input_size: int
            number of characters in the alphabet

        embed_size: int
            size of word embedding

        hidden_size: int
            size of the RNN layer(s)

        use_cuda: bool (default None)
            parameter specifying if GPU is used for computations. If left
            unspecified, GPU will be used if available

        dropout:int
            dropout rate in gru and fc layer

        optimizer_instance: torch.optim object (default torch.optim.Adadelta)
            optimizer to be used for training

        lr: float (default 0.01)
            learning rate for the optimizer

        """
        super(Discriminator, self).__init__()
        self.input_size = input_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.embeddings = nn.Embedding(input_size, embed_size,padding_idx=0)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers=2, bidirectional = True, dropout=dropout,batch_first=True)
        self.hidden2out = nn.Sequential(nn.Linear(2*2*hidden_size, hidden_size), nn.Tanh(),nn.Dropout(p=dropout),nn.Linear(hidden_size, 1),nn.Sigmoid())


        self.use_cuda = use_cuda
        if self.use_cuda is None:
            self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self = self.cuda()
        self.lr = lr
        self.optimizer_instance = optimizer_instance
        self.optimizer = self.optimizer_instance(self.parameters(), lr=lr,
                                                 weight_decay=0.00001)
  

    def init_hidden(self, batch_size):
        if self.use_cuda:
            return torch.zeros(2*2*1, batch_size, self.hidden_size).cuda()
        else:
            return torch.zeros(2*2*1, batch_size, self.hidden_size)

    def classify(self, inp):
        """
        Classifies a sequences.

        Inputs: inp
            - inp: 1 x seq_len

        Returns: out
            - out: 0 to 1  score
        """
        hidden = self.init_hidden(1)
        emb = self.embeddings(inp.view(1,-1))
        self.gru.flatten_parameters()
        _, hidden = self.gru(emb, hidden)                          # 4 x 1 x hidden_dim
        hidden = hidden.permute(1, 0, 2).contiguous()              # batch_size x 4 x hidden_dim
        out = self.hidden2out(hidden.view(-1, 4*self.hidden_size))  # batch_size x 4*hidden_dim
        return out.view(-1).item()

    def getloss(self, inp, length, target):
        """
        Returns Binary Cross Entropy Loss for discriminator.

        Inputs: input, length, target
            - inp: batch_size x seq_len
            - length: batch_size  of  seq_len
            - target: batch_size (binary 1/0)
        """
        batch_size = inp.size()[0]
        hidden = self.init_hidden(batch_size)
        # input dim                                                # batch_size x seq_len  
        emb = self.embeddings(inp)                               # batch_size x seq_len x embed_size
        input_pack = rnn_utils.pack_padded_sequence(emb, length, batch_first=True)
        self.gru.flatten_parameters()
        out, hidden = self.gru(input_pack, hidden)   
        hidden = hidden.permute(1, 0, 2).contiguous()              # batch_size x 4 x hidden_size
        out = self.hidden2out(hidden.view(-1, 4*self.hidden_size))  # batch_size x 4*hidden_size
        loss_fn = nn.BCELoss(reduction='sum')
        return loss_fn(out, target),batch_size

    def train_epochs(self,data_loader,epochs):
        """
        This methods fits the parameters of the model. Training is performed to
        minimize the cross-entropy loss when predicting whether the smiles is ground
        truth or generated

        Parameters
        ----------
        dataloader: object of type DiscriminatorData
            stores information about the data format such alphabet, etc

        epochs: int
            how many iterations of training will be performed

        Returns
        -------
        all_losses: list
            list that stores the values of the loss function (learning curve)
        """    
        all_losses = []
        for epoch in range(0, epochs):
            epoch_loss = 0
            for data,length,label in data_loader:
                self.optimizer.zero_grad()
                loss,batch_size = self.getloss(data,length,label)
                epoch_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            data_loader.reset()
            print('----------Discriminator epoch: %d/%d  loss:%.4f'% (epoch+1,epochs, epoch_loss/len(data_loader.pairs)))
            all_losses.append(epoch_loss/len(data_loader.pairs))
        return all_losses

    def load_model(self, path):
        """
        Loads pretrained parameters from the checkpoint into the model.

        Parameters
        ----------
        path: str
            path to the checkpoint file model will be loaded from.
        """
        weights = torch.load(path)
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