from matplotlib import pyplot as plt
import os
import numpy as np 
import scipy.misc as misc
from torchvision.utils import save_image
from torch.autograd import Variable
import torch

class Visualizer(object):
    def __init__(self, savefolder, imgdim, args):
        self.savefolder = savefolder
        self.save_epoch = 0
        self.name = "default"
        self.plotfolder = os.path.join(savefolder, "plot")
        self.imagedim = imgdim
        self.imagedim.insert(0, -1)
        self.args = args
        if not os.path.isdir(savefolder):
            os.makedirs(savefolder)
        if not os.path.isdir(self.plotfolder):
            os.makedirs(self.plotfolder)

    def visualize(self, imgs, num_rows):
        imgs = imgs.data
        cuname = os.path.join(self.savefolder, self.name + '_current.png')
        epochname = os.path.join(self.savefolder, self.name + '_epoch%d.png' % self.save_epoch)
        save_image(imgs.view(self.imagedim), cuname, nrow=num_rows)
        save_image(imgs.view(self.imagedim), epochname, nrow=num_rows)
        self.save_epoch += 1
        

class ManifoldVisualizer(Visualizer):
    def __init__(self, savefolder, imgdim, args, network):
        super(ManifoldVisualizer, self).__init__(savefolder, imgdim, args)
        self.network = network
        self.name = "manifold"
        self.parts = args.parts

        z_dim = [int(np.prod(self.network.code_dims))]
        self.flat_flag = z_dim[0] >= 2 * self.parts
        self.hierachical_flag = z_dim[0] >= self.parts and len(z_dim) > 1 and z_dim[1] >= 2 
        assert self.flat_flag or self.hierachical_flag
        z_dim.insert(0, self.args.num_rows * self.args.num_rows)

        num_rows = self.args.num_rows
        code_x = torch.FloatTensor(np.linspace(-2, 2, num_rows)).view(1, num_rows).repeat(num_rows, 1).cuda()
        code_y = code_x.t()
        if self.args.ngpus > 0:
            self.z = torch.cuda.FloatTensor(1, z_dim[1]).normal_().repeat(z_dim[0], 1)
            self.code = torch.stack([code_x, code_y], dim=2).view(-1,2).cuda()
        else:
            self.z = torch.FloatTensor(1, z_dim[1]).normal_().repeat(z_dim[0], 1)
            self.code = torch.stack([code_x, code_y], dim=2).view(-1,2)

    def make_code(self, num_rows):
#       z_dim = [int(np.prod(self.network.code_dims))]
        code_x = torch.cuda.FloatTensor(np.linspace(-2, 2, num_rows)).view(1, num_rows).repeat(num_rows, 1)
        code_y = code_x.t()
        if self.args.ngpus > 0:
#            z = torch.cuda.FloatTensor(*z_dim).normal_()
            code = torch.stack([code_x, code_y], dim=2).view(-1,2).cuda()
        else:
#            z = torch.FloatTensor(*z_dim).normal_()
            code = torch.stack([code_x, code_y], dim=2).view(-1,2).cpu()
        return code

    def visualize(self):

        for i in range(self.parts):
            zcode = self.z.clone()
            zcode[:,i*2:i*2+2] = self.code
            imgs = self.network.decode(Variable(zcode)).sigmoid().data
            cuname = os.path.join(self.plotfolder, self.name + '_part%d_current.png' % i)
            epochname = os.path.join(self.savefolder, self.name + '_part%d_epoch%d.png' % (i, self.save_epoch))
            save_image(imgs.view(self.imagedim), cuname, nrow=self.args.num_rows)
            save_image(imgs.view(self.imagedim), epochname, nrow=self.args.num_rows)
        self.save_epoch += 1    

    def visualize_reconstruct(self, imgs):
        num_rows = int(len(imgs) ** 0.5)
        code = self.make_code(num_rows)
        recons,_,_,zcode = self.network(imgs)
        zcode = zcode[1,:].repeat(zcode.size(0), 1)
        recons = recons.sigmoid().data
        cuname = os.path.join(self.plotfolder, 'recon_' + self.name + '_current.png')
        save_image(recons.view(self.imagedim), cuname, nrow=num_rows)
        cuname = os.path.join(self.plotfolder, 'original_' + self.name + '_current.png')
        save_image(imgs.data.view(self.imagedim), cuname, nrow=num_rows)
        for i in range(self.parts):
            zcode[:,i*2:i*2+2] = code
            recons = self.network.decode(zcode).sigmoid().data
            cuname = os.path.join(self.plotfolder, 'recon_' + self.name + '_part%d_current.png' % i)
            save_image(recons.view(self.imagedim), cuname, nrow=num_rows)


    def plot(self, data, name):
        fname = os.path.join(self.plotfolder, name + ".png")
        fig = plt.figure()
        fig.suptitle(name, fontsize=20)
        plt.xlabel('epoch')
        plt.plot(data)
        fig.savefig(fname, dpi=fig.dpi)
        plt.close(fig)

    def mulitplot(self, data, name):
        fname = os.path.join(self.plotfolder, name + ".png")
        fig = plt.figure()
        fig.suptitle(name, fontsize=20)
        plt.xlabel('epoch')
        plt.plot(data)
        plt.legend(["Z" + str(i) for i in range(len(data[0]))],loc='lower right')
        fig.savefig(fname, dpi=fig.dpi)
        plt.close(fig)
