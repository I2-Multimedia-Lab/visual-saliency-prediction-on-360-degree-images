import torch
import time
from model.salnet.basenet import BaseNet
from model.salnet.refinenet import RefineNet
from data.testdataset import MyDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
from PIL import Image
import numpy as np
from scipy import ndimage
import cv2


if __name__ == '__main__':
    with torch.no_grad():
        device = torch.device('cuda')
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        # 加载数据集
        test_data = MyDataset('/home/cxl/Documents/papercode/vis_sal/split3D/test/test.txt',
                               transform=None,)
        # 批量输入
        test_loader = DataLoader(dataset=test_data, batch_size=6, shuffle=False)

        base_model = BaseNet()
        refine_model = RefineNet()
        base_model.to(device)
        refine_model.to(device)

        base_checkpoint = torch.load('/media/cxl/My Passport/LearningMaterials/AboutPapers/PaperCode/vis_sal_weight/'
                                     'basenet_saloss_refinenet_saloss(9+5)_lr_1.3e-05_50/base/20_iter83.pth')
        base_model.load_state_dict(base_checkpoint['basemodel_state_dict'])

        refine_checkpoint = torch.load('/media/cxl/My Passport/LearningMaterials/AboutPapers/PaperCode/vis_sal_weight/'
                                       'basenet_saloss_refinenet_saloss(9+5)_lr_1.3e-05_50/refine/6_iter26.pth')
        refine_model.load_state_dict(refine_checkpoint['refinemodel_state_dict'])

        file_list = ['P81', 'P82', 'P83', 'P84', 'P85', 'P87' ,'P88', 'P89', 'P91', 'P93', 'P94', 'P95', 'P96', 'P97', 'P98']
        for i, (pic, sph) in enumerate(test_loader):

            pic = pic.to(device)
            sph = sph.to(device)

            base_out = base_model(pic)
            refine_out = refine_model(base_out, sph)

            for j in range(6):
                #img = base_out[j][0]*30 * 128 + 31
                img = (refine_out[j][0])*25
                img = img.detach().cpu().numpy()
                img = np.clip(img, 0, 255)
                #img = ndimage.gaussian_filter(img, sigma=3.34) # Gaussian 3.34
                img = np.array(Image.fromarray(img).resize((500, 500)))
                img = Image.fromarray(np.uint8(img),'L')
                img.save("/home/cxl/Documents/papercode/vis_sal/split3D/test/"+ file_list[i] +"/tmp/PatchSaliency{:01d}.png".format(j+1))
                print('saving...' + file_list[i] + ("/tmp/PatchSaliency{:01d}.png").format(j+1))












