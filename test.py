import torch
import time
from model.salnet.basenet import BaseNet
from model.salnet.refinenet import RefineNet
#from model.net2 import Net
from data.traindataset import MyDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
from PIL import Image

if __name__ == '__main__':

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

    with torch.no_grad():
        device = torch.device('cuda')
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        target_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        # 加载数据集
        test_data = MyDataset('/home/cxl/Documents/dataset/ICME2018/test.txt')

        # 批量输入
        test_loader = DataLoader(dataset=test_data, batch_size=10, shuffle=True)

        pic, sal, sph, fix = next(iter(test_loader))

        pic.to(device)
        sph.to(device)

        ToPILImage()(transnorm3(pic[0])).show()
        ToPILImage()(transnorm1(sal[0])).show()

        base_model = BaseNet()
        refine_model = RefineNet()

        base_checkpoint = torch.load('/media/cxl/My Passport/LearningMaterials/AboutPapers/PaperCode/vis_sal_weight/'
                                     'basenet_nssccmse_refinenet_nssccmse_9+5_pre_deepnet_lr_1.3e-05_50/base/14_iter.pth')
        base_model.load_state_dict(base_checkpoint['basemodel_state_dict'])

        refine_checkpoint = torch.load('/medi a/cxl/My Passport/LearningMaterials/AboutPapers/PaperCode/vis_sal_weight/'
                                       'basenet_nssccmse_refinenet_nssccmse_9+5_pre_deepnet_lr_1.3e-05_50/refine/11_iter47.pth')
        refine_model.load_state_dict(refine_checkpoint['refinemodel_state_dict'])


        base_model.eval()
        base_out = base_model(pic)
        ToPILImage()(base_out[0]).show()
        #ToPILImage()(transnorm1(base_out[0])).show()

        refine_model.eval()
        refine_out = refine_model(base_out, sph)
        #ToPILImage()(refine_out[0]).show()
        ToPILImage()(transnorm1(refine_out[0])).show()





