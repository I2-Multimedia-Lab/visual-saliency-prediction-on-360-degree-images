from PIL import Image
from torch.utils.data import Dataset
import torchvision
import cv2
import numpy as np
import torch
from data.preprocessSphInput import preprocessSphInput
from torchvision.transforms import ToPILImage

class MyDataset(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):
        txt = open(txt_path, 'r')
        imgs = []
        for line in txt:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], words[1], words[2], words[3]))
        self.imgs = imgs  # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        stimuli, sal, sph, fix = self.imgs[index]
        # stimulis
        stimulis = Image.open(stimuli).convert('RGB')
        stimulis = stimulis.resize((320, 240), Image.ANTIALIAS) # 像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1
        stimulis = torch.FloatTensor(np.array(stimulis))
        #mean_stimulis = torch.FloatTensor([118.0, 110.0, 100.0])
        mean_stimulis = torch.FloatTensor([118.0, 110.0, 100.0])
        stimulis = stimulis - mean_stimulis.view(1, 1, -1)
        stimulis = (stimulis * 0.0078431372549).permute(2, 0, 1)
        # saliency maps
        sals = Image.open(sal).convert('L')
        sals = sals.resize((320, 240), Image.ANTIALIAS)
        sals = torch.FloatTensor(np.array(sals))
        mean_sals = torch.FloatTensor([31])
        sals = sals - mean_sals
        sals = (sals * 0.0078431372549).unsqueeze(2).permute(2, 0, 1)
        # spherical coords
        sphs = preprocessSphInput(sph)
        sphs = torch.FloatTensor(sphs)
        # fixations
        fix = Image.open(fix).convert('L')
        fix = fix.resize((320, 240))
        fix = torch.FloatTensor(np.array(fix))  # 可能也不需要归一化
        fix = fix.unsqueeze(0)

        # 自定义均值方差时，以下代码没用到，在__getitem__中定义更灵活
        if self.transform is not None:
            stimulis = self.transform(stimulis)  # 在这里做transform，转为tensor等等

        if self.target_transform is not None:
            sals = self.transform(sals)  # 在这里做transform，转为tensor等等


        return stimulis, sals, sphs, fix

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    train_dataset = MyDataset('/home/cxl/Documents/dataset/ICME2018/train.txt',
                              transform=None,
                              target_transform=None)

    def transnorm3(pics):
        a = 100
        b = 110
        c = 118
        # pics = (pics * 255)
        pics[0] = (pics[0]*128 + c)/255
        pics[1] = (pics[1]*128 + b)/255
        pics[2] = (pics[2]*128 + a)/255
        return pics

    def transnorm1(pics):
        a = 31
        # pics = (pics * 255)
        pics = (pics*128 + a) / 255
        return pics

    print(train_dataset[0][0].shape)
    print(train_dataset[0][1])
    print(train_dataset[0][2].shape)
    ToPILImage()(transnorm3(train_dataset[0][0])).show()
    ToPILImage()(train_dataset[0][3]).show()
    # for i in range(240):
    #     print(train_dataset[0][3][0][i])
    # print(train_dataset[0][3])