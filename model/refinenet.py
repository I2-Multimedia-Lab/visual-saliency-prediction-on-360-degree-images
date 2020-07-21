import torch
import torchvision
import torch.nn as nn
from torch.nn import init
class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 5, 1, 2)
        self.act1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(64, 64, 5, 1, 2)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(64, 32, 5, 1, 2)
        self.act4 = nn.ReLU()

        self.conv5 = nn.Conv2d(32, 1, 7, 1, 3)
        self.act5 = nn.ReLU()

        self.deconv1 = nn.ConvTranspose2d(1, 1, 4, 2, 1, output_padding= 0, dilation=1, bias= False)

        self.initWeights()


    def initWeights(self):

        init.normal_(self.conv1.weight, 0, 0.116642368704)
        init.constant_(self.conv1.bias, 0)
        init.normal_(self.conv2.weight, 0, 0.116642368704)
        init.constant_(self.conv2.bias, 0)
        init.normal_(self.conv3.weight, 0, 0.116642368704)
        init.constant_(self.conv3.bias, 0)
        init.normal_(self.conv4.weight, 0, 0.116642368704)
        init.constant_(self.conv4.bias, 0)
        init.normal_(self.conv5.weight, 0, 0.116642368704)
        init.constant_(self.conv5.bias, 0)
        init.normal_(self.deconv1.weight, 0.015625, 0.000001)


    def forward(self, x, y):

        fusion = torch.cat([x, y], dim=1) # [4, 1, 240, 320] [4, 2, 240, 320] cat-> [4, 3, 240, 320]
        fusion = self.maxpool1(self.act1(self.conv1(fusion)))
        fusion = self.act2(self.conv2(fusion))
        fusion = self.act3(self.conv3(fusion))
        fusion = self.act4(self.conv4(fusion))
        fusion = self.act5(self.conv5(fusion))
        fusion = self.deconv1(fusion)

        return fusion

if __name__ == '__main__':
#     net = RefineNet()
#     print(len(list(net.parameters())))
#     for name, parameters in net.named_parameters():
#         print(name, ':', parameters.shape)
#         # print(list(net.parameters())[15])
    img = torch.rand(1, 1, 320, 240)
    sph = torch.rand(1, 2, 320, 240)
    model = RefineNet()
    output = model(img, sph)
    print(output.shape)