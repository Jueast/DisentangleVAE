from matplotlib import pyplot as plt
import os 
import scipy.misc as misc
from torchvision.utils import save_image
class Visualizer(object):
    def __init__(self, savefolder, imgdim):
        self.savefolder = savefolder
        self.save_epoch = 0
        self.name = "default"
        self.imagedim = imgdim
        self.imagedim.insert(0, -1)
        if not os.path.isdir(savefolder):
            os.makedirs(savefolder)

    def visualize(self, imgs, num_rows):
        imgs = imgs.data
        cuname = os.path.join(self.savefolder, 'current.png')
        epochname = os.path.join(self.savefolder, 'epoch%d.png' % self.save_epoch)
        save_image(imgs.view(self.imagedim), cuname, nrow=num_rows)
        save_image(imgs.view(self.imagedim), epochname, nrow=num_rows)
        self.save_epoch += 1
        
