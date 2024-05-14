import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbones import backbone
from .unet_components import *

class BackboneUNet(nn.Module):
    def __init__(self, model_name: str, classes: int):
        """
        Initializes the BackboneUNet model.

        Args:
            model_name (str): Name of the backbone model.
            classes (int): Number of classes for segmentation.
        """
        super(BackboneUNet, self).__init__()
        
        # Initializing backbone and UNet head
        self.backbone = backbone(model_name)
        self.head = get_unet_head(model_name, classes)
        self.dropout = nn.Dropout(p=0)
        self.threshold = 0.5

    def get_results(self, forward_dict, orig_dict):
        """
        Combines forward results with original input dictionary.

        Args:
            forward_dict (dict): Forward pass results.
            orig_dict (dict): Original input dictionary.

        Returns:
            dict: Combined dictionary with segmentation predictions.
        """
        out_dict = {**forward_dict, **orig_dict}

        out_dict['segmentation_preds'] = (forward_dict['logits'].sigmoid() > self.threshold).bool()
        
        return out_dict

    def forward(self, x):
        """
        Forward pass of the BackboneUNet model.

        Args:
            x (tensor): Input tensor.

        Returns:
            dict: Output dictionary with segmentation predictions.
        """
        img = x['data']
        forward_dict = self._forward(img)
        out_dict = self.get_results(forward_dict, x)
        return out_dict

    def _forward(self, x):
        """
        Internal forward pass of the BackboneUNet model.

        Args:
            x (tensor): Input tensor.

        Returns:
            dict: Dictionary containing features and logits.
        """
        backbone_dict = self.backbone(x)

        down1 = backbone_dict['feats_1_map']
        down2 = backbone_dict['feats_2_map']
        down3 = backbone_dict['feats_3_map']
        down4 = backbone_dict['feats_4_map']
        down5 = backbone_dict['feats_last_map']
        
        up1 = self.dropout(self.head.up1(down5, down4))
        up2 = self.dropout(self.head.up2(up1, down3))
        up3 = self.dropout(self.head.up3(up2, down2))
        up4 = self.dropout(self.head.up4(up3, down1))
        
        up = F.interpolate(up4, x.shape[2:], mode='bilinear')
        logits = self.head.out(up)
        
        return {'feats': up4, 'logits': logits}

def get_unet_head(network_name, classes, batch_size=1):
    """
    Retrieves the UNet head based on the network name.

    Args:
        network_name (str): Name of the network.
        classes (int): Number of classes.
        batch_size (int, optional): Batch size. Defaults to 1.

    Returns:
        nn.Module: UNet head module.
    """
    if 'vit' in network_name:
        return UNetHead(256, 128, classes, norm='batch' if batch_size > 1 else 'instance', constant=True)
    elif 'resnet34' in network_name:
        return UNetHead(512, 128, classes, norm='batch' if batch_size > 1 else 'instance')
    else:
        return UNetHead(2048, 128, classes, norm='batch' if batch_size > 1 else 'instance')

class UNetHead(nn.Module):
    def __init__(self, in_channels, ngf, num_classes, norm='batch', constant=False):
        """
        Initializes the UNet head.

        Args:
            in_channels (int): Number of input channels.
            ngf (int): Number of filters in the last layer.
            num_classes (int): Number of classes.
            norm (str, optional): Normalization type. Defaults to 'batch'.
            constant (bool, optional): Whether to use constant input size. Defaults to False.
        """
        super(UNetHead, self).__init__()

        if constant:
            self.up1 = UpInit(in_channels, in_channels, in_channels, True, True)
            self.up2 = Up(2 * in_channels, in_channels, True)
            self.up3 = Up(2 * in_channels, in_channels, True)
            self.up4 = Up(2 * in_channels, in_channels, True)

            self.out = OutConv(in_channels, num_classes)

        else:
            self.up1 = UpInit(in_channels // 2, in_channels, in_channels // 4, True)
            self.up2 = Up(in_channels // 4 * 2, in_channels // 8, True)
            self.up3 = Up(in_channels // 8 * 2, in_channels // 32, True)
            self.up4 = Up(in_channels // 32 * 2, ngf, True)

            self.out = OutConv(ngf, num_classes)
