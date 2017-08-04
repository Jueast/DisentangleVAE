from dataset import Dataset
import  torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.utils.data as tdata
import torch
import numpy as np
from os.path import join, dirname, realpath
import os 
from matplotlib import pyplot as plt

class DspritesDataset(Dataset):

    def __init__(self, batchsize, train=True, classlabel=True):
        Dataset.__init__(self)
        self.root = join(dirname(realpath(__file__)), 'dSprites_data')
        self.name = "dsprites"
        self.range = [0.0, 1.0]
        self.data_dims = [1, 64, 64]
        self.batchsize = batchsize
        self.download()
        self.data = self.process(classlabel)
        self.dataloder = tdata.DataLoader(self.data, self.batchsize, shuffle=True)
        self.iter = iter(self.dataloder)
        self._index = 0

    def next_batch(self):
        image, label = self.iter.next()
        self._index += 1
        if self._index >= len(self.dataloder):
            self.dataloder = tdata.DataLoader(self.data, self.batchsize, shuffle=True)
            self.iter = iter(self.dataloder)
            self._index = 0
            self._epoch += 1 
        return image, label
  
    def __len__(self):
        return len(self.dataloder)

    def dataset_size(self):
        return len(self.dataloder.dataset)

    def index(self):
        return self._index

    def epoch(self):
        return self._epoch

    def image(self, image):
        return np.clip(image, a_min=0.0, a_max=1.0)

    def process(self, classlabel):
        dataset_zip = np.load(join(self.root, "dsprites.npz"), encoding='bytes')
        imgs = torch.from_numpy(dataset_zip['imgs']).float()
        latents_values = torch.from_numpy(dataset_zip['latents_values'])
        latents_classes = torch.from_numpy(dataset_zip['latents_classes'])
        print("Dataset shape: %s" % str(tuple(imgs.size())))
        if classlabel:
            label = latents_classes 
        else:
            label = latents_values
        dataset = tdata.TensorDataset(imgs, label)
        print(imgs[0])
        print(label[0])
        return dataset


    def download(self):
        from six.moves import urllib
        url = "https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz?raw=true"
        if self._check_exists():
            return
        try:
            os.makedirs(self.root)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise
        
        print('Downloading' + url)
        data = urllib.request.urlopen(url)
        filename = "dsprites.npz"
        file_path = os.path.join(self.root, filename)
        with open(file_path, 'wb') as f:
            f.write(data.read())
            f.close()
        print("Done!")
        

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root))