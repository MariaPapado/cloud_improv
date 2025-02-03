import cv2
import numpy as np
import os
from PIL import Image
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A
import albumentations.pytorch
import torchvision.transforms.functional as F
import imutils


class CloudDetection(Dataset):
    def __init__(self, dataset_path, data_list, type='train'):
        self.dataset_path = dataset_path
        self.data_list = data_list
        
        self.type = type


    def __transforms(self, aug, img1, mask):
        if aug:
            img1, mask = imutils.random_fliplr(img1, mask)
            img1, mask = imutils.random_flipud(img1, mask)
        #    img1, img2, mask = imutils.random_rot(img1, img2, mask)

        img1 = imutils.normalize_img(img1)  # imagenet normalization

        img1 = np.transpose(img1, (2, 0, 1))

        return img1, mask

    def __getitem__(self, index):
            
        img1 = Image.open(os.path.join(self.dataset_path, 'images', self.data_list[index]))
        mask = Image.open(os.path.join(self.dataset_path, 'labels', self.data_list[index]))

        img1, mask = np.array(img1), np.array(mask)/255.

        if 'train' in self.type:
            img1, mask = self.__transforms(True, img1, mask)
        else:
            img1, mask = self.__transforms(False, img1, mask)

        #mask = np.expand_dims(mask, 0)

        data_idx = self.data_list[index]
        return np.ascontiguousarray(img1), np.ascontiguousarray(mask)
#        return np.array(img2, dtype=float), np.array(mask, dtype=float), label, data_idx

    def __len__(self):
        return len(self.data_list)
