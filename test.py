
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from utils.dataloader import *

import torch.utils.data as data
import argparse
import logging
from model.unet import *
from PIL import Image
import os
from model.resnet import * 

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--img_size", default=256, type=int)
parser.add_argument('--img_save_path', type=str,default='./snapshot/test/new1/')
parser.add_argument('--weight_save_path', type=str,default='./weight/transfer/G-ci.pth')
parser.add_argument("--data_root", default='./dataset/')
parser.add_argument("--folder", default=["new1"])
parser.add_argument("--label", default={"gender":2,"glass":2,"quality":2}) #0：male/ glass; 1:female / no glass; 2:all data
parser.add_argument("--num_workers", default=16, type=int)
parser.add_argument("--log_path", default='./log/Pix2Pix.log')

opt = parser.parse_args(args=[])

os.makedirs(opt.img_save_path,exist_ok=True)

dataset,data_loader = get_loader(batch_size=opt.batch_size,shuffle=False, label=opt.label,folder=opt.folder,pin_memory=True,img_size=opt.img_size,
                         img_root=opt.data_root,augment=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

G = UNet_transfer(n_channels=3, n_classes=3).to(device)
G.load_state_dict(torch.load(opt.weight_save_path, map_location=torch.device(device)))
C = ResNet(Bottleneck, [3,4,23,3], num_classes=4, num_channels=3,tune=True).to(device)
C.load_state_dict(torch.load('./weight/class4/100D.pth', map_location=torch.device(device)),strict=False)


for i, (img) in enumerate(data_loader):

    print(i)

    img = img.float().to(device)

    with torch.no_grad():
        feature = C(img)
        result = G(img,feature)

    save_image([img[0],result[0]], opt.img_save_path + str(i) + '.png', normalize=True)