# coding: utf-8

import numpy as np
import cv2
import random
import os

"""
    随机挑选CNum张图片，进行按通道计算均值mean和标准差std
    先将像素从0～255归一化至 0-1 再计算
"""

def compute_channel3_mean(train_txt_path):
    #train_txt_path = os.path.join("..", "..", "Data/train.txt")

    CNum = 17290     # 挑选多少图片进行计算

    img_h, img_w = 240, 320
    imgs = np.zeros([img_w, img_h, 3, 1])
    means, stdevs = [], []

    with open(train_txt_path, 'r') as f:
        lines = f.readlines()
        random.shuffle(lines)   # shuffle , 随机挑选图片

        for i in range(CNum):
            img_path = lines[i].rstrip().split()[0]

            img = cv2.imread(img_path)
            img = cv2.resize(img, (img_h, img_w))

            img = img[:, :, :, np.newaxis]
            imgs = np.concatenate((imgs, img), axis=3)
            print(i)
    print(imgs[:,:,:,1])
    #imgs = imgs.astype(np.float32)/255.
    imgs = imgs.astype(np.float32)/255
    print(imgs[:,:,:,1])
    print(imgs.shape)


    for i in range(3):
        pixels = imgs[:,:,i,:].ravel()  # 拉成一行
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    means.reverse() # BGR --> RGB
    stdevs.reverse()

    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))
    print('channel3 transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))


def compute_channel1_mean(train_txt_path):
    # train_txt_path = os.path.join("..", "..", "Data/train.txt")

    CNum = 17290  # 挑选多少图片进行计算

    img_h, img_w = 240, 360
    imgs = np.zeros([img_w, img_h, 1])
    means, stdevs = [], []

    with open(train_txt_path, 'r') as f:
        lines = f.readlines()
        random.shuffle(lines)  # shuffle , 随机挑选图片

        for i in range(CNum):
            img_path = lines[i].rstrip().split()[1]

            img = cv2.imread(img_path, 0)

            img = cv2.resize(img, (img_h, img_w))

            img = img[:, :, np.newaxis]
            imgs = np.concatenate((imgs, img), axis=2)
            print(i)
    print(imgs[:, :, 1])
    # imgs = imgs.astype(np.float32)/255.
    imgs = imgs.astype(np.float32) / 255
    print(imgs[:, :, 1])
    print(imgs.shape)

    pixels = imgs[:, 0, :].ravel()  # 拉成一行
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))

    means.reverse()  # BGR --> RGB
    stdevs.reverse()

    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))
    print('channel1 transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))


if __name__ == '__main__':
    #compute_channel1_mean('/home/cxl/Documents/MATLAB/random_CreateDataset/train.txt')
    compute_channel1_mean('/home/cxl/Documents/MATLAB/random_CreateDataset/test.txt')
    # train_pic --> transforms.Normalize(normMean = [0.4400402, 0.41521043, 0.39547616], normStd = [0.2595092, 0.24837688, 0.2741935])
    # train_pic --> transforms.Normalize(normMean = [112.2103, 105.87864, 100.84661], normStd = [66.174835, 63.336113, 69.91924])
    # train_sal --> transforms.Normalize(normMean = [0.26971766], normStd = [0.15365101])
    # test_pic -->