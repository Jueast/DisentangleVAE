import time
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
class Trainer(object):

    def __init__(self, network, dataset, visualizer,
                 args， optimizer="Adam", lr=1e-3, momentum=0.9, weight_decay=0):
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
        if self.network.name == 'VAEGAN':
            self.lr= lr
            self.weight_decay = weight_decay
            self.momentum = momentum
        else:
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
        if self.network.name != 'VAEGAN':
            self.VAEtrain()
        else:
            self.VAEGANtrain()
            
    def VAEGANtrain(self):
        if optimizer == "Adam":
            self.encoder_optimizer = optim.Adam(self.network.encoder.parameters(),
                                        lr=self.lr,
                                        weight_decay=self.weight_decay)
            self.decoder_optimizer = optim.Adam(self.network.decoder.parameters(),
                                                lr=self.lr,
                                                weight_decay=self.weight_decay)
            self.D_optimizer = optim.Adam(self.network.D.parameters(),
                                          lr=self.lr,
                                          weight_decay = self.weight_decay)
        elif optimizer == "RMSprop":
            self.encoder_optimizer = optim.RMSprop(self.network.encoder.parameters(),
                                        lr=self.lr,
                                        weight_decay=self.weight_decay)
            self.decoder_optimizer = optim.RMSprop(self.network.decoder.parameters(),
                                                lr=self.lr,
                                                weight_decay=self.weight_decay)
            self.D_optimizer = optim.RMSprop(self.network.D.parameters(),
                                          lr=self.lr,
                                          weight_decay = self.weight_decay)
        else:
            self.encoder_optimizer = optim.SGD(self.network.encoder.parameters(),
                                        lr=self.lr,
                                        momentum=self.momentum,
                                        weight_decay=self.weight_decay)
            self.decoder_optimizer = optim.Adam(self.network.decoder.parameters(),
                                                lr=self.lr,
                                                momentum=self.momentum,
                                                weight_decay=self.weight_decay)
            self.D_optimizer = optim.Adam(self.network.D.parameters(),
                                          lr=self.lr,
                                          momentum=self.momentum,
                                          weight_decay = self.weight_decay)
        if self.args.ngpus > 0:
            self.network.cuda()
        self.network.train()
        iteration = 0
        Encoder_Loss_list = []
        Decoder_Loss_list = []
        GAN_Loss_list = []
        BCE_list = []
        KLD_list = []
        MInfo_list = []
        MInfo_split_list = []
        while(iteration < self.maxiters):
            images, _ = self.dataset.next_batch()
            imagesv = Variable(images)
            if iteration == 0:
                spv = imagesv
            if self.cuda:
                imagesv = imagesv.cuda()
            if self.args.ngpus > 0:
                recon_x, mu, logvar, z = nn.parallel.data_parallel(self.network,
                                          images,self.gpuids)
            else:
                recon_x, mu, logvar, z = self.network(imagesv)
            encoder_loss, decoder_loss, GAN_loss, BCE, KLD = self.network.loss(recon_x, imagesv, mu, logvar, z)
            self.encoder_optimizer.zero_grad()
            encoder_loss.backward()
            self.encoder_optimizer.step()
            self.decoder_optimizer.zero_grad()
            decoder_loss.backward()
            self.deconder_optimizer.step()
            minfo, minfo_split = self.network.mutual_info_q(imagesv)
            if iteration % (int(self.args.log_interval/10)) == 0:
                Encoder_Loss_list.append(encoder_loss.data[0])
                Decoder_Loss_list.append(decoder_loss.data[0])
                GAN_Loss_list.append(GAN_loss.data[0])
                Loss_list.append(loss.data[0])
                BCE_list.append(BCE.data[0])
                KLD_list.append(KLD.data[0])
                MInfo_list.append(minfo[0])
                MInfo_split_list.append(minfo_split.numpy())
            if iteration % self.args.log_interval == 0:
                print('#Iter: {}\tTrain Epoch: {}[{}/{}({}%)]\tEncoderLoss:{:6f}\tDecoderLoss{:6f}\n\t\t\tDisLoss:{:6f}\tMInfo:{:6f}'.format(
                    iteration,
                    self.dataset.epoch(),
                    self.dataset.index() * len(images),
                    self.dataset.dataset_size(),
                    int(100. * self.dataset.index() / len(self.dataset)),
                    encoder_loss.data[0],
                    decoder_loss.data[0],
                    GAN_loss.data[0],
                    minfo[0]
                ))
                if self.visualizer.name == "default": 
                    self.visualizer.visualize(recon_x.sigmoid(), self.args.num_rows)
                elif self.visualizer.name == "manifold":
                    self.visualizer.visualize()
                self.visualizer.plot(Encoder_Loss_list, "Encoder_Loss")
                self.visualizer.plot(Decoder_Loss_list, "Decoder_Loss")
                self.visualizer.plot(GAN_Loss_list, "GAN_Loss")
                self.visualizer.plot(BCE_list, "BCE")
                self.visualizer.plot(KLD_list, "KLD or MMD")
                self.visualizer.plot(MInfo_list, "MINFO")
                self.visualizer.mulitplot(MInfo_split_list, "MINFO FOR SPECFIC Z")
                self.visualizer.visualize_reconstruct(spv)
            iteration += 1
           
        
    def VAEtrain(self):
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
            images, _ = self.dataset.next_batch()
            imagesv = Variable(images)
            if iteration == 0:
                spv = imagesv
            if self.cuda:
                imagesv = imagesv.cuda()
            self.optimizer.zero_grad()
            if self.args.ngpus > 0:
                recon_x, mu, logvar, z = nn.parallel.data_parallel(self.network,
                                          images,self.gpuids)
            else:
                recon_x, mu, logvar, z = self.network(imagesv)
            loss, BCE, KLD = self.network.loss(recon_x, imagesv, mu, logvar, z)
            loss.backward()
            self.optimizer.step()
            minfo, minfo_split = self.network.mutual_info_q(imagesv)
            if iteration % (int(self.args.log_interval/10)) == 0:
                Loss_list.append(loss.data[0])
                BCE_list.append(BCE.data[0])
                KLD_list.append(KLD.data[0])
                MInfo_list.append(minfo[0])
                MInfo_split_list.append(minfo_split.numpy())
            if iteration % self.args.log_interval == 0:
                print('#Iter: {}\tTrain Epoch: {}[{}/{}({}%)]\tLoss:{:6f}\tMInfo:{:6f}'.format(
                    iteration,
                    self.dataset.epoch(),
                    self.dataset.index() * len(images),
                    self.dataset.dataset_size(),
                    int(100. * self.dataset.index() / len(self.dataset)),
                    loss.data[0],
                    minfo[0]
                ))
                if self.visualizer.name == "default": 
                    self.visualizer.visualize(recon_x.sigmoid(), self.args.num_rows)
                elif self.visualizer.name == "manifold":
                    self.visualizer.visualize()
                self.visualizer.plot(Loss_list, "Loss")
                self.visualizer.plot(BCE_list, "BCE")
                self.visualizer.plot(KLD_list, "KLD or MMD")
                self.visualizer.plot(MInfo_list, "MINFO")
                self.visualizer.mulitplot(MInfo_split_list, "MINFO FOR SPECFIC Z")
                self.visualizer.visualize_reconstruct(spv)
            iteration += 1

            