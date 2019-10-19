# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F

from unet_parts import *

SCALE = 1

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64 // SCALE)
        self.down1 = down(64 // SCALE, 128 // SCALE)
        self.down2 = down(128 // SCALE, 256 // SCALE)
        self.down3 = down(256 // SCALE, 512 // SCALE)
        self.down4 = down(512 // SCALE, 512 // SCALE)
        self.up1 = up(1024 // SCALE, 256 // SCALE)
        self.up2 = up(512 // SCALE, 128 // SCALE)
        self.up3 = up(256 // SCALE, 64 // SCALE)
        self.up4 = up(128 // SCALE, 64 // SCALE)
        self.outc = outconv(64 // SCALE, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x
