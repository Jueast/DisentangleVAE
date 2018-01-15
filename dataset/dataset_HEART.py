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
    
    def __init__(self, batchsize, name='train/target_normal', recursive=True, data_dims = [1, 32, 32]):
        Dataset.__init__(self)
        data_root = join(dirname(realpath(__file__)), 'HEART_data')
        self.name = "heart"
        self.range = [0.0, 1.0]
        self.data_dims = data_dims
        self.batchsize = batchsize
        data_root = join(data_root, name)
        self.data = self.process(data_root, recursive)    
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
    
    def process(self, data_root, recursive):
        data_list_dir = []
        if recursive:
            def visit_dir(d, stop=False):
                data_list_dir.append(d)
                dir_list = [ join(d, x) for x in os.listdir(d) if os.path.isdir(join(d, x))]
                for sub_d in dir_list:
                    visit_dir(sub_d)
            visit_dir(data_root)
        else:
            data_list_dir.append(data_root)
        print(data_list_dir)
        result = []
        labels = []
        for data_dir in data_list_dir:
            image_list = [ join(data_dir, x) for x in os.listdir(data_dir)
                        if os.path.isfile(join(data_dir, x)) and x.endswith("jpg") ]
            if "normal" in data_dir:
                l = 0
            else:
                l = 1
            print("process %s, label is %d" % (data_dir, l))
            num = 0
            for imgname in image_list:
                num += 1
                im = Image.open(imgname)
                
                im = im.crop((0,0,115,115))
                im = im.filter((ImageFilter.MedianFilter(size=5)))
                im = im.resize(self.data_dims[1:])

                result.append(np.expand_dims(np.asarray(im)/256.0, 0))
                labels.append(l)
            print("process ok! nums: %d" % num)
        imgs = torch.from_numpy(np.asarray(result, dtype=np.float32))
        labels = torch.from_numpy(np.asarray(labels, dtype=np.int32))
        print("Dataset shape: %s" % str(tuple(imgs.size())))       
        return tdata.TensorDataset(imgs, labels)

if __name__ == '__main__':
    batchsize = 25
    heart_data = HeartDataset(batchsize, name='validation')
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

