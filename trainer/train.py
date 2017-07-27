import time
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
class Trainer(object):

    def __init__(self, network, dataset, visualizer, 
                 args, optimizer="Adam", lr=1e-3):
        if args.ngpus > 0:
            self.network = network.cuda()
            self.gpuids = range(args.ngpus)
        else:
            self.network = network
        self.dataset = dataset
        self.visualizer = visualizer
        self.args = args
        self.maxiters = args.maxiters
        self.cuda = args.ngpus > 0
        if optimizer == "Adam":
            self.optimizer = optim.Adam(self.network.parameters(),
                                        lr=lr)

    def train(self):
        self.network.train()
        iteration = 0
        while(iteration < self.maxiters):
            iter_time = time.time()
            images, _ = self.dataset.next_batch()
            imagesv = Variable(images)
            if self.cuda:
                imagesv = imagesv.cuda()
            self.optimizer.zero_grad()
            if self.args.ngpus > 0:
                recon_x, mu, logvar = nn.parallel.data_parallel(self.network,
                                          images,self.gpuids)
            else:
                recon_x, mu, logvar = self.network(imagesv)
            loss = self.network.loss(recon_x, imagesv, mu, logvar)
            loss.backward()
            self.optimizer.step()
            if iteration % self.args.log_interval == 0:
                print('Train Epoch: {}[{}/{}({}%)\t loss:{:6f}'.format(
                    self.dataset.epoch(),
                    self.dataset.index() * len(images),
                    self.dataset.dataset_size(),
                    int(100. * self.dataset.index() / len(self.dataset)),
                    loss.data[0] / len(images)
                ))
                self.visualizer.visualize(recon_x.sigmoid(), 10)
            iteration += 1

            