import time
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch
class Trainer(object):

    def __init__(self, network, dataset, visualizer,
                 args, optimizer="Adam", lr=1e-3, momentum=0.9, weight_decay=0):
        self.model_name = network.name
        self.loss = network.loss
        self.mutual_info_q= network.mutual_info_q
        if args.ngpus > 0:
            self.gpuids = range(args.ngpus)
            self.network = nn.DataParallel(network, self.gpuids).cuda()
            print("running with gpu %s" % str(self.gpuids))
        else:
            self.network = network
        self.dataset = dataset
        self.visualizer = visualizer
        self.args = args
        self.maxiters = args.maxiters
        self.cuda = args.ngpus > 0
        if self.model_name == 'VAEGAN':
            self.lr= lr
            self.weight_decay = weight_decay
            self.momentum = momentum
            self.optimizer = optimizer
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
        if self.model_name != 'VAEGAN':
            self.VAEtrain()
        else:
            self.VAEGANtrain()
            
    def VAEGANtrain(self):
        optimizer = self.optimizer
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
        self.network.train()

        def regForBackward(paras, value):
            for p in paras:
                p.requires_grad=value

        iteration = 0
        Encoder_Loss_list = []
        Decoder_Loss_list = []
        GAN_Loss_list = []
        BCE_list = []
        KLD_list = []
        MInfo_list = []
        MInfo_split_list = []
        clamp_upper = 0.01
        clamp_lower = -0.01
        Encoder_parameters = self.network.encoder.parameters()
        Decoder_parameters = self.network.decoder.parameters()
        D_parameters = self.network.D.parameters()
    
        while(iteration < self.maxiters):
            images, _ = self.dataset.next_batch()
            imagesv = Variable(images)
            one = torch.FloatTensor([1.0])
            mone = one * -1
            
            if self.cuda:
                imagesv = imagesv.cuda()
                one = one.cuda()
                mone = mone.cuda()
            if iteration == 0:
                spv = imagesv
            recon_x, mu, logvar, z = self.network(imagesv)
            encoder_loss, decoder_loss, GAN_loss, BCE, KLD = self.network.loss(recon_x, imagesv, mu, logvar, z)

        
            #regForBackward(Encoder_parameters, True)
            self.encoder_optimizer.zero_grad()
            encoder_loss.backward(one, retain_graph=True)
            self.encoder_optimizer.step()
            #regForBackward(Encoder_parameters, False)

            #regForBackward(Decoder_parameters, True)
            self.decoder_optimizer.zero_grad()
            decoder_loss.backward(one, retain_graph=True)
            self.decoder_optimizer.step()
            #regForBackward(Decoder_parameters, False)

            #regForBackward(D_parameters, True)
            for i in range(5):
                self.D_optimizer.zero_grad()
                GAN_loss = self.network.GAN_loss(imagesv)
                GAN_loss.backward(mone)
                self.D_optimizer.step()
        #    for p in self.network.D.parameters():
        #        p.data.clamp_(clamp_lower, clamp_upper)
            #regForBackward(D_parameters, False)
            minfo, minfo_split = self.mutual_info_q(imagesv)
            if iteration % (int(self.args.log_interval/10)) == 0:
                Encoder_Loss_list.append(encoder_loss.data[0])
                Decoder_Loss_list.append(decoder_loss.data[0])
                GAN_Loss_list.append(GAN_loss.data[0])
                BCE_list.append(BCE.data[0])
                KLD_list.append(KLD.data[0])
                MInfo_list.append(minfo[0])
                MInfo_split_list.append(minfo_split.cpu().numpy())
            if iteration % self.args.log_interval == 0:
                print('#Iter: {}\tTrain Epoch: {}[{}/{}({}%)]\tEncoderLoss:{:6f}\tDecoderLoss:{:6f}\n\t\t\tDisLoss:{:6f}\tMInfo:{:6f}'.format(
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
            if self.cuda:
                imagesv = imagesv.cuda()
            if iteration == 0:
                spv = imagesv
            self.optimizer.zero_grad()
            recon_x, mu, logvar, z = self.network(imagesv)
            loss, BCE, KLD = self.loss(recon_x, imagesv, mu, logvar, z)
            loss.backward()
            self.optimizer.step()
            minfo, minfo_split = self.mutual_info_q(imagesv)
            if iteration % (int(self.args.log_interval/10)) == 0:
                Loss_list.append(loss.data[0])
                BCE_list.append(BCE.data[0])
                KLD_list.append(KLD.data[0])
                MInfo_list.append(minfo[0])
                MInfo_split_list.append(minfo_split.cpu().numpy())
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
            if iteration % (self.args.log_interval * 10) == 0:
                torch.save(self.network, "out/model.pt")
            iteration += 1

            
