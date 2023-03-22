
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch

class ResNet_D(nn.Module):
    "Discriminator ResNet architecture from https://github.com/harryliew/WGAN-QC"
    def __init__(self, size=64, nc=3, nfilter=64, nfilter_max=512, res_ratio=0.1):
        super().__init__()
        s0 = self.s0 = 4
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max
        self.nc = nc

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        nf0 = min(nf, nf_max)
        nf1 = min(nf * 2, nf_max)
        blocks = [
            ResNetBlock(nf0, nf0, bn=False, res_ratio=res_ratio),
            ResNetBlock(nf0, nf1, bn=False, res_ratio=res_ratio)
        ]

        for i in range(1, nlayers+1):
            nf0 = min(nf * 2**i, nf_max)
            nf1 = min(nf * 2**(i+1), nf_max)
            blocks += [
                nn.AvgPool3d(3, stride=2, padding=1),
                ResNetBlock(nf0, nf0, bn=False, res_ratio=res_ratio),
                ResNetBlock(nf0, nf1, bn=False, res_ratio=res_ratio),
            ]

        self.conv_img = nn.Conv3d(nc, 1*nf, 3, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.resnet = nn.Sequential(*blocks)
        self.fc = nn.Linear(self.nf0*s0*s0, 1)
        
        self.final_pooling = nn.MaxPool3d(2, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)

    def forward(self, x):
        batch_size = x.size(0)

        out = self.relu((self.conv_img(x)))
        out = self.resnet(out)
        out = self.final_pooling(out)
        
        out = out.view(batch_size, self.nf0 * self.s0 * self.s0)
        out = self.fc(out)

        return out


class ResNetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, bn=True, res_ratio=0.1):
        super().__init__()
        # Attributes
        self.bn = bn
        self.is_bias = not bn
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden
        self.res_ratio = res_ratio

        # Submodules
        self.conv_0 = nn.Conv3d(self.fin, self.fhidden, 3, stride=1, padding=1, bias=self.is_bias)
        if self.bn:
            self.bn3d_0 = nn.BatchNorm3d(self.fhidden)
        self.conv_1 = nn.Conv3d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=self.is_bias)
        if self.bn:
            self.bn3d_1 = nn.BatchNorm3d(self.fout)
        if self.learned_shortcut:
            self.conv_s = nn.Conv3d(self.fin, self.fout, 1, stride=1, padding=0, bias=False)
            if self.bn:
                self.bn3d_s = nn.BatchNorm3d(self.fout)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(x)
        if self.bn:
            dx = self.bn3d_0(dx)
        dx = self.relu(dx)
        dx = self.conv_1(dx)
        if self.bn:
            dx = self.bn3d_1(dx)
        out = self.relu(x_s + self.res_ratio*dx)
        return out.squeeze()

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
            if self.bn:
                x_s = self.bn3d_s(x_s)
        else:
            x_s = x
        return x_s


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
        diffZ = torch.tensor([x2.size()[4] - x1.size()[4]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, base_factor=32, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.base_factor = base_factor

        self.inc = DoubleConv(n_channels, base_factor)
        self.down1 = Down(base_factor, 2 * base_factor)
        self.down2 = Down(2 * base_factor, 4 * base_factor)
        self.down3 = Down(4 * base_factor, 8 * base_factor // 2)
        factor = 2 if bilinear else 1
        # self.down4 = Down(8 * base_factor, 16 * base_factor // factor)
        # self.up1 = Up(16 * base_factor, 8 * base_factor // factor, bilinear)
        self.up2 = Up(8 * base_factor, 4 * base_factor // factor, bilinear)
        self.up3 = Up(4 * base_factor, 2 * base_factor // factor, bilinear)
        self.up4 = Up(2 * base_factor, base_factor, bilinear)
        self.outc = OutConv(base_factor, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    

def test_resnet():
    f_potential = ResNet_D(16, nc=3)
    sample = torch.rand((10, 3, 16, 16, 16))
    out = f_potential(sample)
    assert out.shape == torch.Size([10, 1])

def test_unet():
    Transport_pushforward = UNet(3, 3)
    sample = torch.rand((10, 3, 16, 16, 16))
    out = Transport_pushforward(sample)
    assert out.shape == torch.Size([10, 3, 16, 16, 16])

test_resnet()
test_unet()
