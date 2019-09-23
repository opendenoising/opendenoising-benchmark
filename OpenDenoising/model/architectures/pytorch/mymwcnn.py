import torch
import torch.nn as nn


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
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

class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        in_batch, in_channel, in_height, in_width = x.size()
        
        out_batch = in_batch
        out_channel = in_channel // 4
        out_height = 2 * in_height
        out_width = 2 * in_width
        
        x1 = x[:, 0 * out_channel: 1 * out_channel, :, :] / 2
        x2 = x[:, 1 * out_channel: 2 * out_channel, :, :] / 2
        x3 = x[:, 2 * out_channel: 3 * out_channel, :, :] / 2
        x4 = x[:, 3 * out_channel: 4 * out_channel, :, :] / 2

        h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

        h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
        h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
        h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
        h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

        return h

class MWCNN(nn.Module):
    """MWCNN Pytorch model. Source code taken from `Github <https://github.com/lpj0/MWCNN_PyTorch>`."""

    def __init__(self, kernel_size=3, n_conv_blocks=4, n_channels=1):
        super(MWCNN, self).__init__()
        self.DWT = DWT()
        self.IWT = IWT()
        
        depth_1_right = [nn.Conv2d(4, 160, padding=(kernel_size//2), kernel_size=kernel_size),
                              nn.ReLU(True)]
        for i in range(n_conv_blocks - 1):
            depth_1_right.append(
                nn.Conv2d(160, 160, padding=(kernel_size//2), kernel_size=kernel_size, bias=False)
            )
            depth_1_right.append(nn.BatchNorm2d(160))
            depth_1_right.append(nn.ReLU(True))
        self.depth_1_right = nn.Sequential(*depth_1_right)

        depth_2_right = [nn.Conv2d(640, 256, padding=(kernel_size//2), kernel_size=kernel_size, bias=False),
                              nn.BatchNorm2d(256), nn.ReLU(True)]
        for i in range(n_conv_blocks - 1):
            depth_2_right.append(
                nn.Conv2d(256, 256, padding=(kernel_size//2), kernel_size=kernel_size, bias=False)
            )
            depth_2_right.append(nn.BatchNorm2d(256))
            depth_2_right.append(nn.ReLU(True))
        self.depth_2_right = nn.Sequential(*depth_2_right)

        depth_3 = [nn.Conv2d(1024, 256, padding=(kernel_size//2), kernel_size=kernel_size, bias=False),
                        nn.BatchNorm2d(256), nn.ReLU(True)]
        for i in range(2 * (n_conv_blocks - 1)):
            depth_3.append(
                nn.Conv2d(256, 256, padding=(kernel_size//2), kernel_size=kernel_size, bias=False)
            )
            depth_3.append(nn.BatchNorm2d(256))
            depth_3.append(nn.ReLU(True))
        depth_3.append(nn.Conv2d(256, 1024, padding=(kernel_size//2), kernel_size=kernel_size, bias=False))
        depth_3.append(nn.BatchNorm2d(1024))
        depth_3.append(nn.ReLU(True))
        self.depth_3 = nn.Sequential(*depth_3)

        depth_2_left = []
        for i in range(n_conv_blocks - 1):
            depth_2_left += [nn.Conv2d(256, 256, padding=(kernel_size//2), kernel_size=kernel_size, bias=False),
                             nn.BatchNorm2d(256), nn.ReLU(True)]
        depth_2_left.append(nn.Conv2d(256, 640, padding=(kernel_size//2), kernel_size=kernel_size, bias=False))
        depth_2_left.append(nn.BatchNorm2d(640))
        depth_2_left.append(nn.ReLU(True))
        self.depth_2_left = nn.Sequential(*depth_2_left)

        depth_1_left = []
        for i in range(n_conv_blocks - 1):
            depth_1_left += [nn.Conv2d(160, 160, padding=(kernel_size//2), kernel_size=kernel_size, bias=False),
                             nn.BatchNorm2d(160), nn.ReLU(True)]
        depth_1_left.append(nn.Conv2d(160, 4, padding=(kernel_size//2), kernel_size=kernel_size, bias=False))
        depth_1_left.append(nn.BatchNorm2d(4))
        depth_1_left.append(nn.ReLU(True))
        self.depth_1_left = nn.Sequential(*depth_1_left)


    def forward(self, x):
        y1 = self.DWT(x)
        y1 = self.depth_1_right(y1)

        y2 = self.DWT(y1)
        y2 = self.depth_2_right(y2)

        y3 = self.DWT(y2)
        y3 = self.depth_3(y3)

        iy3 = self.IWT(y3) + y2
        iy3 = self.depth_2_left(iy3)

        iy2 = self.IWT(iy3) + y1
        iy2 = self.depth_1_left(iy2)

        return self.IWT(iy2) + x
