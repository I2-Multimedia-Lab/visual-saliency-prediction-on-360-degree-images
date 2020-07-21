import torch
import torchvision
import torch.nn as nn
from torch.nn import init
class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=7, stride=1, padding=3)
        self.act1 = nn.ReLU()
        self.lrn = nn.LocalResponseNorm(5, 0.0001, 0.75)  # used in implementation
        self.maxpool1 = nn.MaxPool2d( kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(96, 256, 5, 1, 2)
        self.act2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(256, 512, 3, 1, 1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(512, 512, 5, 1, 2)
        self.act4 = nn.ReLU()

        self.conv5 = nn.Conv2d(512, 512, 5, 1, 2)
        self.act5 = nn.ReLU()

        # self.conv6 = nn.Conv2d(512, 256, 5, 1, 2)
        # self.act6 = nn.ReLU()

        self.conv6 = nn.Conv2d(512, 256, 7, 1, 3)
        self.act6 = nn.ReLU()

        # self.conv7 = nn.Conv2d(256, 128, 7, 1, 3)
        # self.act7 = nn.ReLU()

        self.conv7 = nn.Conv2d(256, 128, 11, 1, 5)
        self.act7 = nn.ReLU()

        self.conv8 = nn.Conv2d(128, 32, 11, 1, 5)
        self.act8 = nn.ReLU()

        self.conv9 = nn.Conv2d(32, 1, 13, 1, 6)
        self.act9 = nn.ReLU()

        self.deconv1 = nn.ConvTranspose2d(1, 1, 8, 4, 2, output_padding= 0, dilation=1, bias= False)

        self.initWeights()


    def initWeights(self):
        init.normal_(self.conv1.weight,0,0.116642)
        init.constant_(self.conv1.bias,0)
        init.normal_(self.conv2.weight,0,0.028867)
        init.constant_(self.conv2.bias,0)
        init.normal_(self.conv3.weight,0,0.029462)
        init.constant_(self.conv3.bias,0)
        init.normal_(self.conv4.weight, 0, 0.0125)
        init.constant_(self.conv4.bias, 0)
        init.normal_(self.conv5.weight, 0, 0.0125)
        init.constant_(self.conv5.bias, 0)
        init.normal_(self.conv6.weight,0,0.0288675134595)
        init.constant_(self.conv6.bias,0)
        init.normal_(self.conv7.weight,0,0.00892857142857)
        init.constant_(self.conv7.bias,0)
        init.normal_(self.conv8.weight,0,0.00803530433167)
        init.constant_(self.conv8.bias,0)
        init.normal_(self.conv9.weight,0,0.0135982073305)
        init.constant_(self.conv9.bias,0)
        init.normal_(self.deconv1.weight,0.015625,0.000001)


    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.lrn(x)
        x = self.maxpool1(x)
        x = self.maxpool2(self.act2(self.conv2(x)))
        x = self.act3(self.conv3(x))
        x = self.act4(self.conv4(x))
        x = self.act5(self.conv5(x))
        x = self.act6(self.conv6(x))
        x = self.act7(self.conv7(x))
        x = self.act8(self.conv8(x))
        x = self.act9(self.conv9(x))
        x = self.deconv1(x)

        return x



if __name__ == '__main__':
    img = torch.rand(1, 3, 320, 240)
    model = BaseNet()
    output = model(img)
    print(output.shape)



# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=96, kernel_size=7, stride=1, padding=3),
#             nn.ReLU(),
#             nn.MaxPool2d( kernel_size=3, stride=2)
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(96, 256, 5, 1, 2),
#             nn.ReLU(),
#             nn.MaxPool2d(3, 2)
#         )
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(256, 512, 3, 1, 1),
#             nn.ReLU()
#         )
#         self.conv4 = nn.Sequential(
#             nn.Conv2d(512, 256, 5, 1, 2),
#             nn.ReLU()
#         )
#         self.conv5 = nn.Sequential(
#             nn.Conv2d(256, 128, 7, 1, 3),
#             nn.ReLU()
#         )
#         self.conv6 = nn.Sequential(
#             nn.Conv2d(128, 32, 11, 1, 5),
#             nn.ReLU()
#         )
#         self.conv7 = nn.Sequential(
#             nn.Conv2d(32, 1, 13, 1, 6),
#             nn.ReLU()
#         )
#         self.deconv1 = nn.Sequential(
#             nn.ConvTranspose2d(1, 1, 8, 4, 2)
#         )
#         self.conv8 = nn.Sequential(
#             nn.Conv2d(1, 32, 5, 1, 2),
#             nn.ReLU(),
#             nn.MaxPool2d(3, 2)
#         )
#         self.conv9 = nn.Sequential(
#             nn.Conv2d(32, 64, 3, 1, 2),
#             nn.ReLU()
#         )
#         self.conv10 = nn.Sequential(
#             nn.Conv2d(64, 32, 5, 1, 2),
#             nn.ReLU()
#         )
#         self.conv11 = nn.Sequential(
#             nn.Conv2d(32, 1, 6, 1, 3), # 论文中卷积核是７ｘ７，我改成了６x６
#             nn.ReLU()
#         )
#         self.deconv2 = nn.Sequential(
#             nn.ConvTranspose2d(1, 1, 4, 2, 1, bias= False),
#         )
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.conv4(x)
#         x = self.conv5(x)
#         x = self.conv6(x)
#         x = self.conv7(x)
#         x = self.deconv1(x)
#         x = self.conv8(x)
#         x = self.conv9(x)
#         x = self.conv10(x)
#         x = self.conv11(x)
#         x = self.deconv2(x)
#         return x


# if __name__ == '__main__':
#     net = Net()
#     print(len(list(net.parameters())))
#     for name, parameters in net.named_parameters():
#         print(name, ':', parameters.shape)
#         # print(list(net.parameters())[15])