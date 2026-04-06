# Import necessary packages
import torch
from torch import nn

# The first discriminator is a simple CNN inspired from PatchGAN (Pix2Pix) which uses 4x4 kernels and stride=2

class Discriminator(nn.Module):
    def __init__(
        self,
        in_chans: int = 2,
        chans: int = 64,
    ):
        super().__init__()

        self.in_chans = in_chans
        self.chans = chans

        self.model = nn.Sequential(
            # Layer 1 (no BatchNorm)
            nn.Conv2d(in_chans, chans, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # Layer 2
            nn.Conv2d(chans, chans * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(chans * 2),
            nn.ReLU(inplace=True),

            # Layer 3
            nn.Conv2d(chans * 2, chans * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(chans * 4),
            nn.ReLU(inplace=True),

            # Layer 4
            nn.Conv2d(chans * 4, chans * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(chans * 8),
            nn.ReLU(inplace=True),

            # Layer 5 (stride = 1)
            nn.Conv2d(chans * 8, chans * 8, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(chans * 8),
            nn.ReLU(inplace=True),

            # Layer 6 (output)
            nn.Conv2d(chans * 8, 1, kernel_size=4, stride=1, padding=1),
            # no sigmoid here as this is done with nn.BCEWithLogitsLoss in the loss functions
        )

    def forward(self, x, y):
        return self.model(torch.cat([x, y], dim=1))


class Discriminator_v2(nn.Module):
    def __init__(self, in_channels=2, base_channels=64):
        super().__init__()

        def conv_block(in_c, out_c, stride=2, norm=True):
            layers = [
                nn.Conv2d(in_c, out_c, kernel_size=4, stride=stride, padding=1)
            ]
            if norm:
                layers.append(nn.InstanceNorm2d(out_c, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *conv_block(in_channels, base_channels, norm=False),

            *conv_block(base_channels, base_channels * 2),

            *conv_block(base_channels * 2, base_channels * 4),

            *conv_block(base_channels * 4, base_channels * 8),

            nn.Conv2d(base_channels * 8, base_channels * 8,
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 8, 1, kernel_size=4, stride=1, padding=0),
            nn.Flatten()  # scalar output
        )

    def forward(self, aliased_img, target_img):
        x = torch.cat([aliased_img, target_img], dim=1)
        return self.model(x)