import torch.nn as nn
import torch.nn.functional as F

class ResnetBlock(nn.Module):
    """Reimplented version of resnet blocks found in CycleGAN implementation"""

    def __init__(self, channels, kernel_size=3, stride=1):
        super(ResnetBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=stride)


        self.conv2 = nn.Conv2d(in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=stride)

        self.adain1 = AdaIN()
        self.adain2 = AdaIN()

    def forward(self, x):
        out = self.conv1(x)
        out = self.adain1(out)
        out = F.ReLU(out)
        out = self.conv2(out)
        out = self.adain2(out)
        out = F.ReLU(out)

        return out + x
