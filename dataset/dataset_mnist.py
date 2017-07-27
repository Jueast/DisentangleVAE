from dataset.dataset import Dataset
import  torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.utils.data as tdata
import numpy as np
from os.path import join, dirname, realpath
from matplotlib import pyplot as plt

class MnistDataset(Dataset):
    
    def __init__(self, batchsize, train=True):
        Dataset.__init__(self)
        data_root = join(dirname(realpath(__file__)), 'MNIST_data')
        self.name = "mnist"
        self.range = [0.0, 1.0]
        self.data_dims = [28, 28, 1]
        self.batchsize = batchsize
        self.data = dsets.MNIST(root=data_root,
                           download=True,
                           train=train,
                           transform=transforms.Compose([
                                transforms.ToTensor()]))
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

if __name__ == '__main__':
    batchsize = 100
    mnist_data = MnistDataset(100)
    while True:
        sample_image, _ = mnist_data.next_batch()
        if mnist_data.index() % 25000 == 0:
            for index in range(9):
                plt.subplot(3, 3, index+1)
                plt.imshow(sample_image[index,0,:,:].numpy(),
                           cmap=plt.get_cmap('Greys'))
            plt.show()

