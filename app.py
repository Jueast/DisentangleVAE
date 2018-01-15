from dataset import *
from model import *
from trainer import *
from visualizer import *
import argparse
import torch
import os
import numpy as np
def get_model(model_name):
    model = torch.load(model_name)
    return model.module.cpu()

def init(model_name, data_dir, batchsize=1):
    model = get_model(model_name)
    dataset = HeartDataset(batchsize, data_dir)
    return model, dataset

def make_cmp_plot(model, img):
    new_img, mu, logvar, z = model(Variable(img))
    new_img = new_img.sigmoid().data.numpy()
    img = img.numpy()
    plt.figure(1)
    plt.imshow(np.squeeze(img))
    plt.figure(2)
    plt.imshow(np.squeeze(new_img))
    plt.show()
