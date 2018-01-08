from dataset import Dataset
from PIL import Image, ImageFilter
import torchvision.transforms as transforms
import torch.utils.data as tdata
import torch
import numpy as np
from os.path import join, dirname, realpath
import os
from matplotlib import pyplot as plt

class HeartDataset(Dataset):
    
    def __init__(self, batchsize, train=True):
        Dataset.__init__(self)
        data_root = join(dirname(realpath(__file__)), 'HEART_data')
        self.name = "heart"
        self.range = [0.0, 1.0]
        self.data_dims = [1, 115, 120]
        self.batchsize = batchsize
        if train:
            data_root = join(data_root, 'train/target_normal')
        else:
            data_root = join(data_root, 'validate/target_normal')
        self.data = self.process(data_root)    
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
    
    def process(self, data_root):
        image_list = [ join(data_root, x) for x in os.listdir(data_root)
                       if os.path.isfile(join(data_root, x)) and x.endswith("jpg") ]
        result = []
        labels = []
        #resize code
        self.data_dims = [1, 32, 32]
        for imgname in image_list:
            
            im = Image.open(imgname)
            
            im = im.crop((0,0,115,115))
            im = im.filter((ImageFilter.MedianFilter(size=5)))
            im = im.resize(self.data_dims[1:])

            result.append(np.expand_dims(np.asarray(im)/256.0, 0))
            labels.append(0)
        imgs = torch.from_numpy(np.asarray(result, dtype=np.float32))
        labels = torch.from_numpy(np.asarray(labels, dtype=np.int32))
        print("Dataset shape: %s" % str(tuple(imgs.size())))
        print(imgs)       
        return tdata.TensorDataset(imgs, labels)

if __name__ == '__main__':
    batchsize = 25
    heart_data = HeartDataset(batchsize)
    print(len(heart_data))
    while True:
        sample_image, _ = heart_data.next_batch()
        if sample_image.shape[0] < 9:
            continue
        if heart_data.index() % 10 == 0:
            for index in range(9):
                plt.subplot(3, 3, index+1)
                plt.imshow(
                    np.moveaxis(sample_image[index,0,:,:].numpy(), 0, -1))
            plt.show()

