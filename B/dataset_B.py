import sys
sys.path.append('./')

from A.dataset import get_loader,montage2d,montage_and_save
import numpy as np
from tqdm import trange
import random
import os
import skimage
from skimage.util import montage as skimage_montage
from PIL import Image
BATCH_SIZE = 128

SPLIT_DICT = {
    "train": "TRAIN",
    "val": "VALIDATION",
    "test": "TEST"
}  


'''This script is just for having a general view of the dataset'''

'''Dataset PathMNIST (pathmnist)
    Number of datapoints: 89996
    Root location: /home/three/.medmnist
    Split: train
    Task: multi-class
    Number of channels: 3
    Meaning of labels: {'0': 'adipose', '1': 'background', '2': 'debris', '3': 'lymphocytes', '4': 'mucus', '5': 'smooth muscle', '6': 'normal colon mucosa', '7': 'cancer-associated stroma', '8': 'colorectal adenocarcinoma epithelium'}
    Number of samples: {'train': 89996, 'val': 10004, 'test': 7180}
    Description: The PathMNIST is based on a prior study for predicting survival from colorectal cancer 
    histology slides, providing a dataset (NCT-CRC-HE-100K) of 100,000 non-overlapping image patches from hematoxylin
      & eosin stained histological images, and a test dataset (CRC-VAL-HE-7K) of 7,180 image patches from 
      a different clinical center. The dataset is comprised of 9 types of tissues, resulting in a multi-class classification task. 
      We resize the source images of 3×224×224 into 3×28×28, and split NCT-CRC-HE-100K into training and validation set with a ratio of 9:1. The CRC-VAL-HE-7K is treated as the test set.'''




data_flag = 'pathmnist'
input_root = 'Datasets/'

n_channels = 3
n_classes = 9

dataset = np.load(os.path.join(input_root, "{}.npz".format(data_flag)))


print(dataset.files)
print(dataset['train_labels'])

train_images = dataset['train_images']
train_loader = get_loader(dataset=train_images, batch_size=BATCH_SIZE)

montage_and_save(data_flag,train_images,n_channels)
