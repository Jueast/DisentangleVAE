from model.abstract_VAE import VAE
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable, grad
from scipy.stats import norm

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        #m.weight.data.normal_(0.0, clamp_upper-clamp_lower)
        nn.init.xavier_uniform(m.weight)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform(m.weight)

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


class Discriminator(nn.Module):

    def __init__(self, input_dims, hidden=400, activacation="lrelu", batchnorm=False):
        super(Discriminator, self).__init__()
        if activacation == "lrelu":
            self.act = nn.LeakyReLU()
        else:
            self.act = nn.ReLU()
            
        self.nx = int(np.prod(input_dims))
        self.main = nn.Sequential(
            nn.Linear(self.nx, hidden),
            self.act,
            nn.Linear(hidden, hidden),
            self.act,
            nn.Linear(hidden, hidden),
            self.act,
            nn.Linear(hidden, 1)
        )

    def forward(self, input):
        input = input.view(input.size(0), -1)
        out = self.main(input).mean(0)
        return out
    
class VAEGAN(VAE):

    def __init__(self, input_dims, code_dims, beta=1.0, gamma=0.75,
                 hidden=400, activacation="lrelu",
                 decoder="Bernoulli", batchnorm=False):

        super(VAEGAN, self).__init__(input_dims, code_dims)
        self.name = "VAEGAN"
        self.nx = int(np.prod(input_dims))
        self.nz = int(np.prod(code_dims))
        self.beta = beta
        self.gamma = gamma
        if activacation == "lrelu":
            self.act = nn.LeakyReLU()
        else:
            self.act = nn.ReLU()
        if decoder == "Bernoulli":
            self.reconstruct_loss = StableBCELoss()
        else:
            self.reconstruct_loss = nn.MSELoss()

        self.encoder = nn.ModuleList([EncodeLayer(self.nx, hidden, code_dims[1], batchnorm, activacation)]) 
        self.decoder_list = nn.ModuleList([])
        self.decoder = nn.ModuleList([])
        for i in range(code_dims[0]-1):
            el = EncodeLayer(hidden, hidden, code_dims[1], batchnorm, activacation)
            dl = DecodeLayer(hidden, hidden, code_dims[1], batchnorm, activacation)
            self.encoder.append(el)
            self.decoder_list.append(dl)
        self.decoder.append(self.decoder_list)
        self.fc1 = nn.Linear(code_dims[1], hidden)
        self.fc2 = nn.Linear(hidden, self.nx)
        self.decoder.append(self.fc1)
        self.decoder.append(self.fc2)
        
        self.D = Discriminator(input_dims, hidden, activacation, batchnorm)
        
        self.D.apply(weights_init)
        
    def encode(self, x):
        h = x.view(x.size(0), -1)
        mu_list = []
        logvar_list = []
        for fc in self.encoder:
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
        for z, fc in zip(zcode[1:], self.decoder_list):
            h = fc(h, z)
        return self.fc2(h)

    def forward(self, x):
        mu, logvar = self.encode(x.view(x.size(0), -1))
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar, z
    def prior_loss(self, mu, logvar, z):
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5) / mu.size(0)
        return KLD
    def match_loss(self, recon_x, x):
        x = x.view(x.size(0), -1)
        BCE = self.reconstruct_loss(recon_x, x) / x.size(0)
        return BCE 
    def GAN_loss(self, x):
        x = x.view(x.size(0), -1)
        if isinstance(x, torch.cuda.FloatTensor):
            eps = torch.cuda.FloatTensor(x.size(0), self.nz).normal_()
        else:
            eps = torch.FloatTensor(x.size(0), self.nz).normal_()
        alpha = torch.FloatTensor(x.size(0), 1).uniform_(0,1)
        alpha = alpha.expand(x.size(0), x.size(1))
        recon_pz = self.decode(Variable(eps))
        interpolates = alpha * x.data + (1-alpha) * recon_pz.data
        interpolates = Variable(interpolates, requires_grad=True)
        D_interpolates = self.D(interpolates)
        gradients = grad(D_interpolates, interpolates,create_graph=True)[0]
        slopes = torch.sum(gradients ** 2, 1).sqrt()
        gradient_penalty = (torch.mean(slopes - 1.) ** 2)
        return self.D(x) - self.D(recon_pz) - 10 * gradient_penalty
    def encoder_loss(self, recon_x, x, mu, logvar, z):
        BCE = self.match_loss(recon_x, x)
        KLD = self.prior_loss(mu, logvar, z)
        return BCE + self.beta * KLD
 
    def decoder_loss(self, recon_x, x, mu, logvar, z):
        BCE = self.match_loss(recon_x, x)
        GAN_loss = - self.D(reconx)
        return  BCE + GAN_loss * self.gamma

    def loss(self, recon_x, x, mu, logvar, z):
        BCE = self.match_loss(recon_x, x)
        KLD = self.prior_loss(mu, logvar, z)
        GAN_loss = self.GAN_loss(x)
        return BCE + self.beta * KLD, BCE + GAN_loss * self.gamma , GAN_loss, BCE, KLD
    
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


