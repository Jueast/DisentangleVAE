from dataset import *
from model import *
from trainer import *
from visualizer import *
import argparse
import torch
import os
import numpy as np
from torchvision.utils import make_grid
def get_model(model_name):
    model = torch.load(model_name)
    return model.module.cpu()

def init(model_name, data_dir, batchsize=1):
    model = get_model(model_name)
    dataset = HeartDataset(batchsize, data_dir)
    return model, dataset

def make_cmp_plot(model, dataset):
    imgs,_ = dataset.next_batch()
    imgs_figure = make_grid(imgs, nrow=int((imgs.size(0) ** 0.5))).numpy()
    new_imgs, mu, logvar, z = model(Variable(imgs))
    new_img_figure = make_grid(new_imgs.sigmoid().data, nrow=int((new_imgs.size(0) ** 0.5))).numpy()
    plt.figure(1)
    plt.imshow(np.moveaxis(imgs_figure, 0, -1))
    plt.figure(2)
    plt.imshow(np.moveaxis(new_img_figure, 0, -1))
    plt.show()

def show_distri(model, dataset):
    imgs, labels = dataset.total_data()
    new_imgs, mu, logvar, z = model(Variable(imgs))
    print("z' shape: %s" % (str(tuple(z.size()))))
    l = int(z.size(1))
    z = z.data
    for i in range(0, l, 2):
        fig = plt.figure()
        plt.scatter(x=z[:,i].numpy(), y=z[:,i+1].numpy(), s=2, c=labels.numpy(),  alpha=0.5)
        plt.xlabel('z%d' % i)
        plt.ylabel('z%d' % (i+1))
    plt.show()
        