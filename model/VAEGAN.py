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



