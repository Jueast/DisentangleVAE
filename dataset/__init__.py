try:    # Works for python 3
    from dataset.dataset import *
    from dataset.dataset_mnist import MnistDataset
    from dataset.dataset_SVHN import SVHNDataset
    from dataset.dataset_dsprites import DspritesDataset
    from dataset.dataset_HEART import HeartDataset
except: # Works for python 2
    from dataset import *
    from dataset_mnist import MnistDataset
    from dataset_SVHN import SVHNDataset
    from dataset_dsprites import DspritesDataset
    from dataset_HEART import HeartDataset