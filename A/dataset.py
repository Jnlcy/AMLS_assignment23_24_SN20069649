
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import cv2
import os
from tqdm import tqdm
import skimage
from skimage.util import montage as skimage_montage
from PIL import Image


datapath = 'Datasets/pneumoniamnist.npz'

'''"label": {
            "0": "normal",
            "1": "pneumonia"
        },
        "n_channels": 1,
        "n_samples": {
            "train": 4708,
            "val": 524,
            "test": 624
        },'''
  
n_channels = 1
n_classes = 2

# load dataset

def get_loader(dataset, batch_size):
    total_size = len(dataset)
    print('Size', total_size)
    index_generator = shuffle_iterator(range(total_size))
    while True:
        data = []
        for _ in range(batch_size):
            idx = next(index_generator)
            data.append(dataset[idx])
        yield dataset._collate_fn(data)


def shuffle_iterator(iterator):
    # iterator should have limited size
    index = list(iterator)
    total_size = len(index)
    i = 0
    random.shuffle(index)
    while True:
        yield index[i]
        i += 1
        if i >= total_size:
            i = 0
            random.shuffle(index)

def montage2d(imgs, n_channels, sel):
    sel_img = imgs[sel]

    # version 0.20.0 changes the kwarg `multichannel` to `channel_axis`
    if skimage.__version__ >= "0.20.0":
        montage_arr = skimage_montage(
            sel_img, channel_axis=3 if n_channels == 3 else None)
    else:
        montage_arr = skimage_montage(sel_img, multichannel=(n_channels == 3))
    montage_img = Image.fromarray(montage_arr)

    return montage_img

'''------- Copied exactly from the MEDMNIST repo: https://github.com/MedMNIST/MedMNIST/blob/main/examples/dataset_without_pytorch.py#L132-------'''
npz = np.load(datapath)
# print(npz_file.files)
print(npz['train_labels'])
BATCH_SIZE = 128
train_images = npz['train_images']
train_loader = get_loader(dataset=train_images, batch_size=BATCH_SIZE)

length = 20
n_sel = length * length
sel = np.random.choice(train_images.shape[0], size=n_sel, replace=False)

montage_img = montage2d(imgs=train_images,
                        n_channels=n_channels,
                        sel=sel)



save_folder = 'save_folder'
if save_folder is not None:
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    montage_img.save(os.path.join(save_folder,
                                    f"task_A_train_montage.jpg"))


