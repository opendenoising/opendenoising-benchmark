import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


def default_conv1(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels,out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, groups=1)


def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    #print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h



class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)

class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)



class MWCNN(nn.Module):
    """MWCNN Pytorch model. Source code taken from `Github <https://github.com/lpj0/MWCNN_PyTorch>`.
    """
    def __init__(self, args, conv=common.default_conv):
        super(BSR, self).__init__()
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        self.scale_idx = 0

        act = nn.ReLU(True)

        self.DWT = common.DWT()
        self.IWT = common.IWT()

        n = 3
        m_head = [common.BBlock(conv, 4, 160, 3, act=act)]
        d_l1 = []
        for _ in range(n):
            d_l1.append(common.BBlock(conv, 160, 160, 3, act=act))

        d_l2 = [common.BBlock(conv, 640, n_feats * 4, 3, act=act)]
        for _ in range(n):
            d_l2.append(common.BBlock(conv, n_feats * 4, n_feats * 4, 3, act=act))

        pro_l3 = [common.BBlock(conv, n_feats * 16, n_feats * 4, 3, act=act)]
        for _ in range(n*2):
            pro_l3.append(common.BBlock(conv, n_feats * 4, n_feats * 4, 3, act=act))
        pro_l3.append(common.BBlock(conv, n_feats * 4, n_feats * 16, 3, act=act))

        i_l2 = []
        for _ in range(n):
            i_l2.append(common.BBlock(conv, n_feats * 4, n_feats * 4, 3, act=act))
        i_l2.append(common.BBlock(conv, n_feats * 4,640, 3, act=act))

        i_l1 = []
        for _ in range(n):
            i_l1.append((common.BBlock(conv,160, 160, 3, act=act)))

        m_tail = [conv(160, 4, 3)]

        self.head = nn.Sequential(*m_head)
        self.d_l2 = nn.Sequential(*d_l2)
        self.d_l1 = nn.Sequential(*d_l1)
        self.pro_l3 = nn.Sequential(*pro_l3)
        self.i_l2 = nn.Sequential(*i_l2)
        self.i_l1 = nn.Sequential(*i_l1)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):

        x1 = self.d_l1(self.head(self.DWT(x)))
        x2 = self.d_l2(self.DWT(x1))
        # x3 = self.d_l2(self.DWT(x2))
        x_ = self.IWT(self.pro_l3(self.DWT(x2))) + x2
        x_ = self.IWT(self.i_l2(x_)) + x1

        # x = self.i_l0(x) + x0
        x = self.IWT(self.tail(self.i_l1(x_))) + x
        # x = self.add_mean(x)

        return x

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx