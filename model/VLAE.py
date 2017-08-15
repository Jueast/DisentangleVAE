from model.abstract_VAE import VAE
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from scipy.stats import norm

class StableBCELoss(nn.modules.Module):
    def __init__(self):
        super(StableBCELoss, self).__init__()
    def forward(self, input, target):
        neg_abs = - input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.sum()

class CNNEncodeLayer(nn.Module):
    def __init__(self, input, output, zdim, batchnorm, activacation):
        super(CNNEncodeLayer, self).__init__()
        if activacation == "lrelu":
            self.act = nn.LeakyReLU()
        else:
            self.act = nn.ReLU()
        if batchnorm:
            main = nn.Sequential(
                nn.Conv2d(input, output, kernel=4, stride=2, padding=1),
                nn.BatchNorm2d(output),
                self.act,
            )
        else:
            main = nn.Sequential(
                nn.Conv2d(input, output, kernel=4, stride=2, padding=1),
                self.act,
            )
        self.conv = nn.Conv2d(output, 1, kernel=1, stride=1, padding=0)
        print ("Not implemented now...")
        return 

class EncodeLayer(nn.Module):
    def __init__(self, input, output, zdim, batchnorm, activacation):
        super(EncodeLayer, self).__init__()
        if activacation == "lrelu":
            self.act = nn.LeakyReLU()
        else:
            self.act = nn.ReLU()
        if batchnorm:
            main = nn.Sequential(
                nn.Linear(input, output),
                nn.BatchNorm1d(output),
                self.act,
            )
        else:
            main = nn.Sequential(
                nn.Linear(input, output),
                self.act,
            )
        self.main = main
        self.fc1 = nn.Linear(output, zdim)
        self.fc2 = nn.Linear(output, zdim)
    def forward(self, x):
        h = self.main(x)
        return self.main(x),self.fc1(h), self.fc2(h)


class DecodeLayer(nn.Module):
    def __init__(self, input, output, zdim, batchnorm, activacation):
        super(DecodeLayer, self).__init__()
        
        if activacation == "lrelu":
            self.act = nn.LeakyReLU()
        else:
            self.act = nn.ReLU()
        if input == 0:
            input = output
            self.fc = nn.Linear(zdim, input)
        else:
            self.fc = nn.Linear(zdim, input)
            input *= 2
        if batchnorm:
            main = nn.Sequential(
                nn.Linear(input, output),
                nn.BatchNorm1d(output),
                self.act,
            )
        else:
            main = nn.Sequential(
                nn.Linear(input, output),
                self.act,
            )
        self.main = main

    def forward(self, input, z):
        if input is None:
            input = self.act(self.fc(z))
        else:
            input = torch.cat([input, self.act(self.fc(z))], 1)
        return self.main(input)

class VLAE(VAE):

    def __init__(self, input_dims, code_dims, beta=1.0,
                 hidden=400, activacation="lrelu",
                 decoder="Bernoulli", batchnorm=False):

        super(VLAE, self).__init__(input_dims, code_dims)
        self.name = "VLAE"
        self.nx = int(np.prod(input_dims))
        self.nz = int(np.prod(code_dims))
        self.beta = beta
        if activacation == "lrelu":
            self.act = nn.LeakyReLU()
        else:
            self.act = nn.ReLU()
        if decoder == "Bernoulli":
            self.reconstruct_loss = StableBCELoss()
        else:
            self.reconstruct_loss = nn.MSELoss()

        self.encode_layers = nn.ModuleList([EncodeLayer(self.nx, hidden, code_dims[1], batchnorm, activacation)]) 
        self.decode_layers = nn.ModuleList([])
        for i in range(code_dims[0]-1):
            el = EncodeLayer(hidden, hidden, code_dims[1], batchnorm, activacation)
            dl = DecodeLayer(hidden, hidden, code_dims[1], batchnorm, activacation)
            self.encode_layers.append(el)
            self.decode_layers.append(dl)
    
        self.fc1 = nn.Linear(code_dims[1], hidden)
        self.fc2 = nn.Linear(hidden, self.nx)

    def encode(self, x):
        h = x.view(x.size(0), -1)
        mu_list = []
        logvar_list = []
        for fc in self.encode_layers:
            h, mu, logvar = fc(h)
            mu_list.append(mu)
            logvar_list.append(logvar)
        return torch.cat(mu_list, dim=1), torch.cat(logvar_list, dim=1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if isinstance(mu, torch.cuda.FloatTensor):
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
#        eps[:,-2:-1] = (eps[:,-2:-1] - mu.data[:,-2:-1]) / std.data[:,-2:-1]
        eps = Variable(eps)
        return eps.mul(std).add_(mu) 
    
    def decode(self, z):
        zcode = list(torch.chunk(z, self.code_dims[0], dim=1))[::-1]
        h = self.act(self.fc1(zcode[0]))
        for z, fc in zip(zcode[1:], self.decode_layers):
            h = fc(h, z)
        return self.fc2(h)

    def forward(self, x):
        mu, logvar = self.encode(x.view(x.size(0), -1))
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar, z

    def loss(self, recon_x, x, mu, logvar, z):
        x = x.view(x.size(0), -1)
        BCE = self.reconstruct_loss(recon_x, x) / x.size(0)
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5) / x.size(0)
        return BCE + self.beta * KLD, BCE, KLD

    def mutual_info_q(self, x):
        mu, logvar = self.encode(x.view(x.size(0), -1))
        z = self.reparametrize(mu, logvar)
        l = z.size(0)
        z = z.repeat(l, 1, 1)
        mu = mu.unsqueeze(2).repeat(1,1,l).transpose(1,2)
        logvar = logvar.unsqueeze(2).repeat(1,1,l).transpose(1,2)
        p_matrix =  ( - torch.sum((z - mu) ** 2  / logvar.exp(), dim=2) / 2.0 - 0.5 * torch.sum(logvar, dim=2)).exp_()
        p_split_matrix = (- (z - mu) ** 2  / logvar.exp() / 2.0 - 0.5 * logvar ).exp_()
        p_split_vector = torch.sum(p_split_matrix, dim=1)
        p_vector =  torch.sum(p_matrix, dim=1)
        I = torch.FloatTensor([np.log(l)])
        I_split = torch.FloatTensor([np.log(l)] * int(z.size(2)))
        for i in range(l):
            I += (p_matrix[i][i].log() - p_vector[i].log()).data / l
            I_split += (p_split_matrix[i][i].log() - p_split_vector[i].log()).data / l
        # q(z_i) is not independent..
        # assert np.allclose(I.numpy(), np.sum(I_split.numpy()))
        return I, I_split


class MMDVLAE(VLAE):
    def compute_kernel(self, x, y):
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        tiled_x = x.unsqueeze(1).repeat(1, y_size, 1)
        tiled_y = y.unsqueeze(0).repeat(x_size, 1, 1)
        return ((-(tiled_x - tiled_y) ** 2).mean(dim=2) / float(dim)).exp_()
    
    def compute_mmd(self, x, y, sigma_sqr=1.0):
        x_kernel = self.compute_kernel(x, x)
        y_kernel = self.compute_kernel(y, y)
        xy_kernel = self.compute_kernel(x, y)
        return torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)
    def loss(self, recon_x, x, mu, logvar, z):
        x = x.view(x.size(0), -1)
        BCE = self.reconstruct_loss(recon_x, x) / x.size(0)
        
        true_samples = Variable(torch.FloatTensor(x.size(0), self.nz).normal_())
        MMD = self.compute_mmd(true_samples, z)
        return BCE + self.beta *  MMD , BCE, MMD