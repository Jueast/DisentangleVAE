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
        code_dims = self.network.code_dims[:]
        z_dim = code_dims
        z_dim.insert(0, 15 * 15)
        if self.args.ngpus > 0:
            self.z = torch.cuda.FloatTensor(*z_dim).normal_()
        else:
            self.z = torch.FloatTensor(*z_dim).normal_()
    
    def visualize(self, imgs, num_rows, parts=3):
        code_dims = self.network.code_dims[:]
        flat_flag = code_dims[0] >= 2 * parts
        hierachical_flag = code_dims[0] >= parts and len(code_dims) > 1 and code_dims[1] >= 2 
        assert flat_flag or hierachical_flag

        code_x = torch.linspace(-2, 2, steps=num_rows).view(1, num_rows).repeat(num_rows, 1)
        code_y = code_x.t()
        code = torch.stack([code_x, code_y], dim=2).view(-1,2)
        z = self.z
  #      z_dim = code_dims
  #      z_dim.insert(0, num_rows * num_rows)
 #       if self.args.ngpus > 0:
  #          z = torch.cuda.FloatTensor(*z_dim).normal_()
  #      else:
  #          z = torch.FloatTensor(*z_dim).normal_()
        for i in range(parts):
            if flat_flag:
                zcode = z.clone()
                zcode[:,parts*2:parts*2+2] = code
                imgs = self.network.decode(Variable(zcode)).sigmoid().data
                cuname = os.path.join(self.savefolder, self.name + '_part%d_current.png' % i)
                epochname = os.path.join(self.savefolder, self.name + '_part%d_epoch%d.png' % (i, self.save_epoch))
                save_image(imgs.view(self.imagedim), cuname, nrow=num_rows)
                save_image(imgs.view(self.imagedim), epochname, nrow=num_rows)

            else:
                print("TODO: Hierachical")
                exit("-1")
        self.save_epoch += 1    

        

