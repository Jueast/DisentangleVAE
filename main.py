
import matplotlib
matplotlib.use('Agg')

import os
from dataset import *
from model import *
from trainer import *
from visualizer import *
import argparse
import torch
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--ngpus', type=int, default=4)
	parser.add_argument('--dataset', type=str, default='MNIST')
	parser.add_argument('--optim', type=str, default='Adam')
	parser.add_argument('--model', type=str, default='VAE')
	parser.add_argument('--savefolder', type=str, default='out')
	parser.add_argument('--hidden', type=int, default=400)
	parser.add_argument('--dimz', type=int, default=30)
	parser.add_argument('--hlayers', type=int, default=3)
	parser.add_argument('--beta', type=float, default=1.0)
	parser.add_argument('--gamma', type=float, default=0.75)
	parser.add_argument('--num_rows', type=int, default=10)
	parser.add_argument('--visualizer', type=str, default="manifold")
	parser.add_argument('--lr',type=float, default=1e-3)
	parser.add_argument('--l2',type=float, default=2.5e-5)
	parser.add_argument('--log_interval', type=int, default=100)
	parser.add_argument('--maxiters', type=int, default=100000)
	parser.add_argument('--batchsize', type=int, default=100)
	parser.add_argument('--parts', type=int, default=2)
	parser.add_argument('--seed', type=int, default=115414)
	parser.add_argument('--visible_gpus', type=str, default="3,6,7")
	args = parser.parse_args()


	if args.dataset == 'MNIST':
	    dataset = MnistDataset(args.batchsize)
	elif args.dataset == 'SVHN':
	    dataset = SVHNDataset(args.batchsize)
	elif args.dataset == 'DSPRITES':
	    dataset = DspritesDataset(args.batchsize)
	elif args.dataset == 'HEART':
	    dataset = HeartDataset(args.batchsize)
	else:
	    print("Unknown dataset")
	    exit(-1)
	torch.set_default_tensor_type('torch.cuda.FloatTensor')
	os.environ['CUDA_VISIBLE_DEVICES'] = args.visible_gpus
	torch.cuda.manual_seed_all(args.seed)
	torch.manual_seed(args.seed)
		
	if args.model == 'VAE':
	    network = NaiveVAE(dataset.data_dims, [args.dimz], hidden=args.hidden)
	elif args.model == 'betaVAE':
	    network = BetaVAE(dataset.data_dims, [args.dimz],
			      hidden=args.hidden, beta=args.beta)
	elif args.model == 'MMDVAE':
	    network = MMDVAE(dataset.data_dims, [args.dimz],
			      hidden=args.hidden, beta=args.beta)
	elif args.model == 'VLAE':
	    network = VLAE(dataset.data_dims, [args.hlayers, int(args.dimz / args.hlayers)], hidden=args.hidden, beta=args.beta)
	elif args.model == 'CNNVLAE':
	    network = CNNVLAE(dataset.data_dims, [args.hlayers, int(args.dimz / args.hlayers)], hidden=args.hidden, beta=args.beta)
	elif args.model == 'MMDVLAE':
	    network = MMDVLAE(dataset.data_dims, [args.hlayers, int(args.dimz / args.hlayers)], hidden=args.hidden, beta=args.beta)
	elif args.model == 'VAEGAN':
	    network = VAEGAN(dataset.data_dims, [args.hlayers, int(args.dimz / args.hlayers)], hidden=args.hidden, beta=args.beta, gamma=args.gamma)
	else:
	    print("Unknown model")
	    exit(-1)
	print(network)
	if args.visualizer == 'manifold':
	    visualizer = ManifoldVisualizer(args.savefolder, dataset.data_dims, args, network)
	else:
	    visualizer = Visualizer(args.savefolder, dataset.data_dims, args)
	trainer = Trainer(network, dataset, visualizer, args, lr=args.lr, weight_decay=args.l2)
	trainer.train()
