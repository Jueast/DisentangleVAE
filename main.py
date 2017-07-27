import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--ngpus', type=int, default=4)
parser.add_argument('--dataset', type=str, default='MNIST')
parser.add_argument('--model', type=str, default='VAE')
parser.add_argument('--savefolder', type=str, default='out')
parser.add_argument('--dimz', type=int, default=30)
parser.add_argument('--lr',type=float, default=1e-3)
parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--maxiters', type=int, default=100000)
parser.add_argument('--use_gui', dest='use_gui', action='store_true',
                    help='Display the results with a GUI window')
parser.add_argument('--batchsize', type=int, default=100)

args = parser.parse_args()

import matplotlib
if not args.use_gui:
    matplotlib.use('Agg')
else:
    from matplotlib import pyplot as plt
    plt.ion()
    plt.show()

import os
from dataset import *
from model import *
from trainer import *
from visualizer import *

if args.dataset == 'MNIST':
    dataset = MnistDataset(args.batchsize)
else:
    print("Unknown dataset")
    exit(-1)

if args.model == 'VAE':
    network = NaiveVAE(dataset.data_dims, [args.dimz])
else:
    print("Unknown model")
    exit(-1)

visualizer = Visualizer(args.savefolder, dataset.data_dims)
trainer = Trainer(network, dataset, visualizer, args, lr=args.lr)
trainer.train()