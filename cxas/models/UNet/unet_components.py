import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """Double Convolution Block: (convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        """
        Initialize DoubleConv module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            mid_channels (int, optional): Number of mid channels. Defaults to None.
        """
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward pass of DoubleConv module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.double_conv(x)

class Conv(nn.Module):
    """Convolution Block: (convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels):
        """
        Initialize Conv module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Forward pass of Conv module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        """
        Initialize Down module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        """
        Forward pass of Down module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        """
        Initialize Up module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            bilinear (bool, optional): Whether to use bilinear interpolation. Defaults to True.
        """
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        self.up = F.interpolate
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        """
        Forward pass of Up module.

        Args:
            x1 (torch.Tensor): Input tensor 1.
            x2 (torch.Tensor): Input tensor 2.

        Returns:
            torch.Tensor: Output tensor.
        """
        x1 = self.up(x1, x2.shape[2:], mode='bilinear')
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UpInit(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, temp_channels, in_channels, out_channels, bilinear=True, hax=False):
        """
        Initialize UpInit module.

        Args:
            temp_channels (int): Number of temporary channels.
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            bilinear (bool, optional): Whether to use bilinear interpolation. Defaults to True.
            hax (bool, optional): Whether to apply special conditions. Defaults to False.
        """
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        self.up = F.interpolate
        self.conv1 = Conv(in_channels, temp_channels)
        if hax:
            self.conv2 = DoubleConv(in_channels+temp_channels, out_channels, in_channels // 2)
        else:
            self.conv2 = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        """
        Forward pass of UpInit module.

        Args:
            x1 (torch.Tensor): Input tensor 1.
            x2 (torch.Tensor): Input tensor 2.

        Returns:
            torch.Tensor: Output tensor.
        """
        x1 = self.conv1(x1)
        x1 = self.up(x1, x2.shape[2:], mode='bilinear')
        x = torch.cat([x2, x1], dim=1)
        return self.conv2(x)


class OutConv(nn.Module):
    """Output Convolution Block.

    This block consists of a single convolutional layer without activation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """Forward pass through the output convolution block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.conv(x)

