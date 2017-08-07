import time
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
class Trainer(object):

    def __init__(self, network, dataset, visualizer, 
                 args, optimizer="Adam", lr=1e-3, momentum=0.9, weight_decay=0):
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
                                        lr=lr,
                                        weight_decay=weight_decay)
        elif optimizer == "RMSprop":
            self.optimizer = optim.RMSprop(self.network.parameters(),
                                           lr=lr,
                                           weight_decay=weight_decay)
        else:
            self.optimizer = optim.SGD(self.network.parameters(),
                                           lr=lr,
                                           momentum=momentum,
                                           weight_decay=weight_decay)
    def train(self):
        if self.args.ngpus > 0:
            self.network.cuda()
        self.network.train()
        iteration = 0
        Loss_list = []
        BCE_list = []
        KLD_list = []
        MInfo_list = []
        MInfo_split_list = []
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
            loss, BCE, KLD = self.network.loss(recon_x, imagesv, mu, logvar)
            loss.backward()
            self.optimizer.step()
            minfo, minfo_split = self.network.mutual_info_q(imagesv)
            if iteration % (int(self.args.log_interval/10)) == 0:
                Loss_list.append(loss.data[0] / len(images))
                BCE_list.append(BCE.data[0] / len(images))
                KLD_list.append(KLD.data[0] / len(images))
                MInfo_list.append(minfo[0])
                MInfo_split_list.append(minfo_split.numpy())
            if iteration % self.args.log_interval == 0:
                print('#Iter: {}\tTrain Epoch: {}[{}/{}({}%)]\tLoss:{:6f}\tMInfo:{:6f}'.format(
                    iteration,
                    self.dataset.epoch(),
                    self.dataset.index() * len(images),
                    self.dataset.dataset_size(),
                    int(100. * self.dataset.index() / len(self.dataset)),
                    loss.data[0] / len(images),
                    minfo[0]
                ))
                if self.visualizer.name == "default": 
                    self.visualizer.visualize(recon_x.sigmoid(), self.args.num_rows)
                elif self.visualizer.name == "manifold":
                    self.visualizer.visualize()
                self.visualizer.plot(Loss_list, "Loss")
                self.visualizer.plot(BCE_list, "BCE")
                self.visualizer.plot(KLD_list, "KLD")
                self.visualizer.plot(MInfo_list, "MINFO")
                self.visualizer.mulitplot(MInfo_split_list, "MINFO FOR SPECFIC Z")
            iteration += 1

            