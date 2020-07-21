import torch
import time
import os
#from model.salnet import Net
#from model.net2 import Net
from model.salnet.basenet import BaseNet
from model.salnet.refinenet import RefineNet
from data.traindataset import MyDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from losses.saloss import Saloss
import matplotlib.pyplot as plt
import itertools

if __name__ == '__main__':

    # print train start time
    print('start time -->', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    # define super parameters
    BACT_SIZE = 4
    #LR = 1.30208333333e-07
    LR = 1.3e-06
    EPOCH = 50

    # load datset
    train_dataset = MyDataset('/home/cxl/Documents/dataset/ICME2018/train.txt',
                          transform=None,
                          target_transform=None)

    # 讲train划分成 train(16000) 和 val(1290)
    train_db, val_db = torch.utils.data.random_split(train_dataset, [16000, 1290])
    # batch input
    train_loader = DataLoader(dataset=train_db, batch_size=BACT_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset=val_db, batch_size=BACT_SIZE, shuffle=True, num_workers=4)

    base_model = BaseNet()
    refine_model = RefineNet()
    # load the weights of deepnet
    # base_model.load_state_dict(torch.load('utils/caffemodel_to_pytorch/deepnet/deepnet_params.pth'))

    # finetune, 迁移学习，加载部分预训练权重，basenet的前三层, refinenet的前两层
    # base_model.load_state_dict(torch.load('utils/caffemodel_to_pytorch/salnet/salnet360_base_top3_params.pth'))
    # refine_model.load_state_dict(torch.load('utils/caffemodel_to_pytorch/salnet/salnet360_refine_top2_params.pth'))

    # salnet360_base_top3_params = torch.load('utils/caffemodel_to_pytorch/salnet/salnet360_base_top3_params.pth')
    # base_model_dict = base_model.state_dict()
    # base_state_dict = {k: v for k, v in salnet360_base_top3_params.items() if k in base_model_dict.keys()}
    # base_model_dict.update(base_state_dict)
    # base_model.load_state_dict(base_model_dict)
    #
    # salnet360_refine_top2_params = torch.load('utils/caffemodel_to_pytorch/salnet/salnet360_refine_top2_params.pth')
    # refine_model_dict = refine_model.state_dict()
    # refine_state_dict = {k: v for k, v in salnet360_refine_top2_params.items() if k in refine_model_dict.keys()}
    # refine_model_dict.update(refine_state_dict)
    # refine_model.load_state_dict(refine_model_dict)

    device = torch.device('cuda:0')
    base_model.to(device) # 将model放到cuda
    refine_model.to(device)

    # 加载断点，预训练权重

    base_lossfc = Saloss()
    refine_lossfc = Saloss()
    # nss_lossfc = NSS()
    # cc_lossfc = CC()
    # kld_lossfc = torch.nn.KLDivLoss()

    base_optim = torch.optim.Adam(base_model.parameters(), lr=LR, weight_decay=5e-07)
    refine_optim = torch.optim.Adam(refine_model.parameters(), lr=LR, weight_decay=5e-07)
    # base_optim = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-07, nesterov=True) # 效果不好，容易在局部最小值里震荡

    path = 'basenet_saloss_refinenet_saloss_9+5_lr_' + str(LR) + '_' + str(EPOCH)
    model_path = '/media/cxl/My Passport/LearningMaterials/AboutPapers/PaperCode/vis_sal_weight/'+path

    os.makedirs(model_path)
    os.makedirs(model_path + '/base')
    os.makedirs(model_path + '/refine')

    # visualize loss value in the browser by tensorboard
    writer_train = SummaryWriter('runs_salnet/' + path)
    writer_val = SummaryWriter('runs_salnet/' + path)
    iter = 0# tensorboard 记录val打印次数
    
    '''
    ========================= train ================================
    '''
    
    for epoch in range(EPOCH):
        base_model.train()
        refine_model.train()
        for i, (pic, sal, sph, fix) in enumerate(train_loader):
            # forward
            pic = pic.to(device) # 将pic放入cuda
            sal = sal.to(device) # 将sal放入cuda
            sph = sph.to(device)
            fix = fix.to(device)

            base_out = base_model(pic)
            refine_out = refine_model(base_out, sph)
            
            base_loss = base_lossfc(base_out, sal, fix)
            refine_loss = refine_lossfc(refine_out, sal, fix)

            base_optim.zero_grad()
            base_loss.backward(retain_graph=True) # 保留由于自定义loss等的复杂性，需要一次forward()，多个不同loss的backward()来累积同一个网络的grad,来更新参数。于是，若在当前backward()后，不执行forward() 而是执行另一个backward()，需要在当前backward()时，指定保留计算图，backward(retain_graph)。
            base_optim.step()
            
            refine_optim.zero_grad()
            refine_loss.backward()
            refine_optim.step()
            
            # 每迭代500次打印一次loss值，共8*500=4000张图片
            if i % 1000 == 0:

                ################################  base info ###################################
                print('BASE Train EPOCH:{}, iteration:{}, train_base_loss:{:.8f}'.format(epoch, iter, base_loss))
                writer_train.add_scalar('Loss/train_base_loss', base_loss, iter) # 将loss值显示在tensorboard
                base_save_path = model_path + '/'+ 'base' + '/' + str(epoch) + '_' + 'iter' + str(iter) + '.pth'
                # 以字典的形式保存训练时的多个信息，包括 net, net.parameters, base_optim, train_loss, epoch, 同时可以用来断点训练
                base_state = {'base_model': base_model,
                              'basemodel_state_dict':base_model.state_dict(),
                              'base_optim_state_dict': base_optim.state_dict(),
                              'epoch': epoch,
                              'base_loss':base_loss.detach().item()}
                torch.save(base_state, base_save_path)
                
                ################################  refine info ###################################
                print('REFINE Train EPOCH:{}, iteration:{}, train_refine_loss:{:.8f}'.format(epoch, iter, refine_loss))
                writer_train.add_scalar('Loss/train_refine_loss', refine_loss, iter)  # 将loss值显示在tensorboard
                refine_save_path = model_path + '/' + 'refine' + '/' + str(epoch) + '_' + 'iter' + str(iter) + '.pth'
                # 以字典的形式保存训练时的多个信息，包括 net, net.parameters, refine_optim, train_loss, epoch, 同时可以用来断点训练
                refine_state = {'refine_model': refine_model,
                                'refinemodel_state_dict': refine_model.state_dict(),
                                'refine_optim_state_dict': refine_optim.state_dict(),
                                'epoch': epoch,
                                'refine_loss': refine_loss.detach().item()}
                torch.save(refine_state, refine_save_path)

                iter = iter + 1
                
        '''
                    ========================= val ================================
        '''
        base_model.eval()
        refine_model.eval()
        # 每一次epoch做一次交叉验证, 全val
        val_base_loss = 0
        val_refine_loss = 0
        with torch.no_grad():
            for j, (pic, sal, sph, fix) in enumerate(val_loader):

                pic = pic.to(device)  # 将pic放入cuda
                sal = sal.to(device)  # 将sal放入cuda
                sph = sph.to(device)
                fix = fix.to(device)

                base_out = base_model(pic)
                refine_out = refine_model(base_out, sph)

                val_base_loss += base_lossfc(base_out, sal, fix).detach().item()
                val_refine_loss += refine_lossfc(refine_out, sal, fix).detach().item()
            # 打印val_loss的信息
            print('EPOCH:{}, val_base_loss:{:.8f}'.format(epoch, val_base_loss / len(val_loader)))
            print('EPOCH:{}, val_refine_loss:{:.8f}'.format(epoch, val_refine_loss / len(val_loader)))

            # 将train_loss和val_loss值显示在tensorboard,　在一张图上显示
            writer_val.add_scalar('Loss/val_base_loss', val_base_loss / len(val_loader), epoch)
            writer_val.add_scalar('Loss/val_refine_loss', val_refine_loss / len(val_loader), epoch)

    writer_train.close()
    writer_val.close()

    # print train end time
    print('end time -->', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))



