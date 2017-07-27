# Refer to pytorch/examples/VAE

from model.abstract_VAE import VAE
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

class StableBCELoss(nn.modules.Module):
       def __init__(self):
             super(StableBCELoss, self).__init__()
       def forward(self, input, target):
             neg_abs = - input.abs()
             loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
             return loss.sum()

class NaiveVAE(VAE):

    def __init__(self, input_dims, code_dims,
                 hidden=400, activacation="lrelu",
                 decoder="Bernoulli"):
        super(NaiveVAE, self).__init__(input_dims,
                                        code_dims)
        self.nx = int(np.prod(input_dims))
        self.nz = int(np.prod(code_dims))
        
        if activacation == "lrelu":
            self.act = nn.LeakyReLU()
        else:
            self.act = nn.ReLU()
        
        if decoder == "Bernoulli":
            self.reconstruct_loss = nn.BCELoss()
        else:
            self.reconstruct_loss = nn.MSELoss()
        # encoding part
        self.fc1 = nn.Linear(self.nx, hidden)
        # mu and sigma
        self.fc21 = nn.Linear(hidden, self.nz)
        self.fc22 = nn.Linear(hidden, self.nz)

        # decoding part
        self.fc3 = nn.Linear(self.nz, hidden)
        self.fc4 = nn.Linear(hidden, self.nx)

    def encode(self, x):
        x = x.view(x.size(0), -1)
        h1 = self.act(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if isinstance(mu, torch.cuda.FloatTensor):
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = self.act(self.fc3(z))
        return self.fc4(h3).sigmoid()
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(x.size(0), -1))
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss(self, recon_x, x, mu, logvar):
        x = x.view(x.size(0), -1)
        BCE = self.reconstruct_loss(recon_x, x)
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5)
        return BCE + KLD

