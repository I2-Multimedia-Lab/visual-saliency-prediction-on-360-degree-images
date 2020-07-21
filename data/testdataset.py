from PIL import Image
from torch.utils.data import Dataset
from data.preprocessSphInput import preprocessSphInput
import torchvision
import cv2
import numpy as np
import torch

class MyDataset(Dataset):
    def __init__(self, txt_path, transform=None):
        txt = open(txt_path, 'r')
        imgs = []
        for line in txt:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], words[1]))

        self.imgs = imgs  # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
        self.transform = transform

    def __getitem__(self, index):
        stimuli, sph= self.imgs[index]
        # stimuli
        stimulis = Image.open(stimuli).convert('RGB')
        stimulis = stimulis.resize((320, 240), Image.ANTIALIAS)# 像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1
        stimulis = torch.FloatTensor(np.array(stimulis))
        #mean_stimulis = torch.FloatTensor([118.0, 110.0, 100.0])
        mean_stimulis = torch.FloatTensor([118.0, 110.0, 100.0])
        stimulis = stimulis - mean_stimulis.view(1, 1, -1)
        stimulis = (stimulis * 0.0078431372549).permute(2, 0, 1)

        # spherical coords
        sphs = preprocessSphInput(sph)
        sphs = torch.FloatTensor(sphs)

        if self.transform is not None:
            stimulis = self.transform(stimulis)  # 在这里做transform，转为tensor等等


        return stimulis, sphs

    def __len__(self):
        return len(self.imgs)