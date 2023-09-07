import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat



class DoubleConv(nn.Module):
    """(convolution => [IN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv_down = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(mid_channels),
            # nn.BatchNorm2d(mid_channels),#BatchNorm2d
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            # nn.BatchNorm2d(out_channels),#BatchNorm2d
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv_down(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

# class Down_IN(nn.Module):
#     """Downscaling with maxpool then double conv"""

#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.maxpool_conv = nn.Sequential(
#             nn.MaxPool2d(2),
#             DoubleConv_IN(in_channels, out_channels)
#         )

#     def forward(self, x):
#         return self.maxpool_conv(x)



class Up_noskip(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
            # self.conv = DoubleConv(in_channels//2, out_channels)

    def forward(self, x):
        x = self.up(x)

        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)




class fusion(nn.Module):
    """Upscaling then double conv"""

    def __init__(self):
        super().__init__()

        self.down = Down(in_channels=2048, out_channels=1024)
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024,kernel_size=2, stride=2),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
                                  

    def forward(self, x7,x6,feature):
        x = self.down(torch.cat((x6,feature),dim=1))
        x = torch.cat((x,x7),dim=1)
        x = self.conv(x)

        return x

class UNet_transfer(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_transfer, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 512)
        self.down5 = Down(512, 1024)
        self.down6 = Down(1024, 1024)

        self.fuse = fusion()
        
        self.up1 = Up_noskip(1024, 512, bilinear)
        self.up2 = Up_noskip(512, 256, bilinear)
        self.up3 = Up_noskip(256, 128, bilinear)
        self.up4 = Up_noskip(128, 64, bilinear)
        self.up5 = Up_noskip(64, 64, bilinear)

        self.outc = OutConv(64, n_classes)
        self.sig = nn.Sigmoid()

        self.conv = DoubleConv(2048, 1024)

    def forward(self, x, feature):

        x = self.inc(x) #64, 256, 256
        x = self.down1(x) #128, 128, 128
        x = self.down2(x) #256, 64, 64
        x = self.down3(x) #512, 32, 32
        x = self.down4(x) #512, 16, 16
        x6 = self.down5(x) #1024, 8, 8
        x7 = self.down6(x6) #1024, 4, 4


        feature = self.conv(feature)

        x = self.fuse(x7,x6,feature) # (1024, 4, 4) (1024, 8, 8) (1024, 8, 8) -> (512,16,16)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        x = self.outc(x)
        x = self.sig(x)

        return x


