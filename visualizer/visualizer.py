from matplotlib import pyplot as plt
import os 
import scipy.misc as misc
from torchvision.utils import save_image
from torch.autograd import Variable
import torch
class Visualizer(object):
    def __init__(self, savefolder, imgdim, args):
        self.savefolder = savefolder
        self.save_epoch = 0
        self.name = "default"
        self.imagedim = imgdim
        self.imagedim.insert(0, -1)
        self.args = args
        if not os.path.isdir(savefolder):
            os.makedirs(savefolder)

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

        z_dim = self.network.code_dims[:]
        self.flat_flag = z_dim[0] >= 2 * self.parts
        self.hierachical_flag = z_dim[0] >= self.parts and len(z_dim) > 1 and z_dim[1] >= 2 
        assert self.flat_flag or self.hierachical_flag
        z_dim.insert(0, self.args.num_rows * self.args.num_rows)

        num_rows = self.args.num_rows
        code_x = torch.linspace(-2, 2, steps=num_rows).view(1, num_rows).repeat(num_rows, 1)
        code_y = code_x.t()
        if self.args.ngpus > 0:
            self.z = torch.cuda.FloatTensor(*z_dim).normal_()
            self.code = torch.stack([code_x, code_y], dim=2).view(-1,2).cuda()
        else:
            self.z = torch.FloatTensor(*z_dim).normal_()
            self.code = torch.stack([code_x, code_y], dim=2).view(-1,2)


    def visualize(self):

        for i in range(self.parts):
            if self.flat_flag:
                zcode = self.z.clone()
                zcode[:,self.parts*2:self.parts*2+2] = self.code
                imgs = self.network.decode(Variable(zcode)).sigmoid().data
                cuname = os.path.join(self.savefolder, self.name + '_part%d_current.png' % i)
                epochname = os.path.join(self.savefolder, self.name + '_part%d_epoch%d.png' % (i, self.save_epoch))
                save_image(imgs.view(self.imagedim), cuname, nrow=self.args.num_rows)
                save_image(imgs.view(self.imagedim), epochname, nrow=self.args.num_rows)

            else:
                print("TODO: Hierachical")
                exit("-1")
        self.save_epoch += 1    

        

