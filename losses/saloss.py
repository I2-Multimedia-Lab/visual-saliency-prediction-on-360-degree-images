import torch
from torch import Tensor as T
from torch.nn import functional as F
import numpy as np

EPSILON = np.finfo('float').eps

class Saloss(torch.nn.Module):
    def __init__(self):
        super(Saloss, self).__init__()

    def forward(self, output, target, fix):

        train_err = KL_div(output, target)
        #train_err = KL_div(output, target)
        return train_err


def KL_div(output, target):
    # output = output/output.max()
    # target = target/target.max()
    # output = output / output.sum()
    # target = target / target.sum()
    # a = target * T.log(target / (output + 1e-50) + 1e-50)
    # b = output * T.log(output / (target + 1e-50) + 1e-50)
    # return (a.sum() + b.sum()) / (2.)
    output = output / output.sum()
    target = target / target.sum()
    #kl = F.kl_div(output, target)
    #kl = F.kl_div((output+EPSILON).log(), target)
    kl = target * torch.log(target/(output+EPSILON) + EPSILON)
    kl = kl.sum()
    return kl

def CC(output, target):
    output = (output - output.mean()) / output.std()
    target = (target - target.mean()) / target.std()
    num = (output - output.mean()) * (target - target.mean())
    out_squre = (output - output.mean()) ** 2
    tar_squre = (target - target.mean()) ** 2
    cc = num.sum() / (torch.sqrt(out_squre.sum() * tar_squre.sum()))
    return cc


def NSS(output, fix):

    fixationMap = fix > 0 # 0.5
    output = (output - output.mean()) / output.std()
    return output[fixationMap].mean()