# dataloader for fine-tuning   用于test
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
import torch.utils.data as data
import numpy as np
from PIL import ImageEnhance, Image, ImageOps
import random
import os
import glob
import xlrd

def cv_random_flip(img, label):

    flip_flag = random.randint(0, 2)

    if flip_flag == 2:
        img = ImageOps.mirror(img)
        label = ImageOps.mirror(label)

    return img, label

def randomCrop(image, label):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region)

def randomRotation(image, label):
    rotate_time = random.randint(0, 3)
    image = np.rot90(image, rotate_time).copy()
    label = np.rot90(label, rotate_time).copy()
    return image, label

def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image

def randomGaussian(img, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return img

def randomPeper(img):
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):
        randX = random.randint(0, img.shape[0] - 1)
        randY = random.randint(0, img.shape[1] - 1)
        if random.randint(0, 1) == 0:
            img[randX, randY] = 0
        else:
            img[randX, randY] = 1
    return img

class Dataset(data.Dataset):
    def __init__(self,img_size,folder,label,image_root,augment=False):
        
        self.dataset = folder
        self.img = []
      
        for id in self.dataset:

            path = image_root + str(id) + '/'
                
            img = [path + f for f in os.listdir(path) if ((f.endswith('.jpg'))or(f.endswith('.jpeg')))]
           
            for i,path in enumerate(img):
                self.img.append(path)
                    
            
        self.augment = augment
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size,img_size))
        ])
        self.gt_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size,img_size))
        ])
        

    def __getitem__(self, index):

        img = Image.open(self.img[index])
        print(self.img[index])
        img = img.convert("RGB")
        

        img = self.img_transform(img)
       

        return img

    def __len__(self):
        return int(len(self.img))

def get_loader(batch_size, shuffle,folder, label,pin_memory=True,img_size=256, img_root='./dataset/',
               augment=False):
    dataset = Dataset(img_size=img_size, image_root=img_root,label=label,folder=folder,augment=augment)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                                  pin_memory=pin_memory,num_workers=0)#多进程：num_workers=8
    return dataset,data_loader
