import argparse
import array
import json
import numpy as np
import scipy
import torch
import scipy.misc
import torch

from PIL import Image

def preprocessSphInput(sphericalCoordsFileName):
    binaryData = open(sphericalCoordsFileName, 'rb')
    (width, height) = 320, 240
    sizeArr = array.array('H')
    sizeArr.fromfile(binaryData, 1)
    sphCoordsArr = array.array('f')
    sphCoordsArr.fromfile(binaryData, 2*sizeArr[0]*sizeArr[0])
    sphCoordsArr = np.array(sphCoordsArr).reshape(2, sizeArr[0], sizeArr[0])
    sphCoordsArr[1] %= 2*np.pi
    sphCoordsArr[1] -= np.pi
    sphCoordsArr[1] /= np.pi
    sphCoordsArr[0] /= (np.pi/2)
    resizedArr = np.zeros((2, height, width))
    resizedArr[0] = scipy.misc.imresize(sphCoordsArr[0], (height, width), interp='cubic', mode='F')
    resizedArr[1] = scipy.misc.imresize(sphCoordsArr[1], (height, width), interp='cubic', mode='F')
    return resizedArr

if __name__ == '__main__':

    sph = preprocessSphInput('../tmp/PatchSC2.bin')
    sph = torch.FloatTensor(sph)
    # sph = torch.from_numpy(sph)
    # img = Image.open('/home/cxl/Documents/MATLAB/random_CreateDataset/Train_Pictures/P1_900x450_Patch00.jpg').convert('L')
    # img = img.resize((320, 240), Image.ANTIALIAS)
    # img = np.array(img, dtype=float)
    # #img = torch.FloatTensor(np.array(img))
    # img = torch.from_numpy(img).unsqueeze(2).permute(2, 0, 1)
    # sph_img = torch.cat([img, sph], dim=0)
    # print(type(img))
    print(sph)
    print(sph.shape)
