import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict

# Define a shorthand for Batch Normalization
BatchNorm = nn.BatchNorm2d

class Backbone(nn.Module):
    """
    Backbone module for extracting features from pre-trained networks.
    """

    def __init__(self, network_name):
        """
        Initializes the Backbone module.

        Args:
            network_name (str): Name of the pre-trained network.
        """
        super(Backbone, self).__init__()
        network_name = network_name.split('_')[1].lower()
        self.network_name = network_name
        self.backbone = self._get_backbone(network_name)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

    def _get_backbone(self, network_name):
        """
        Gets the backbone network architecture.

        Args:
            network_name (str): Name of the pre-trained network.

        Returns:
            torch.nn.Sequential: Backbone network architecture.
        """
        if network_name == 'vgg16':
            full_net = getattr(models, network_name)()
            features = list(full_net.features)[:30]

            net = nn.Sequential(*features)

            self.classifier = nn.Sequential(*list(full_net.classifier)[:5])

        elif 'resnet' in network_name:
            full_net = getattr(models, network_name)()

            features = [
                ('layer0', torch.nn.Sequential(*[full_net.conv1,
                                                  full_net.bn1,
                                                  full_net.relu,
                                                  full_net.maxpool])),
                ('layer1', full_net.layer1),
                ('layer2', full_net.layer2),
                ('layer3', full_net.layer3),
                ('layer4', full_net.layer4),
            ]

            self.inplanes = full_net.inplanes

            net = nn.Sequential(OrderedDict(features))
        else:
            raise NotImplementedError('{} not implemented as BACKBONE Network'.format(network_name))

        return net

    def _make_layer(self, block, inplanes, planes, blocks, stride=1, dilation=1, new_level=True, residual=True):
        """
        Helper function to create a layer in the network.

        Args:
            block (torch.nn.Module): Basic building block of the layer.
            inplanes (int): Number of input channels.
            planes (int): Number of output channels.
            blocks (int): Number of blocks in the layer.
            stride (int, optional): Stride for the convolutional layers. Defaults to 1.
            dilation (int, optional): Dilation rate for the convolutional layers. Defaults to 1.
            new_level (bool, optional): Flag to indicate if it's a new level. Defaults to True.
            residual (bool, optional): Flag to indicate if residual connections are used. Defaults to True.

        Returns:
            torch.nn.Sequential: Sequential layer containing the specified blocks.
        """
        assert dilation == 1 or dilation % 2 == 0
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = list()
        layers.append(block(
            inplanes, planes, stride, downsample,
            dilation=(1, 1) if dilation == 1 else (
                dilation // 2 if new_level else dilation, dilation),
            residual=residual))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, residual=residual,
                                dilation=(dilation, dilation)))

        return nn.Sequential(*layers)

    def insert_dilations(self, features, dilation_size):
        """
        Inserts dilations into the convolutional layers.

        Args:
            features (torch.nn.Module): Backbone features.
            dilation_size (int): Size of dilation to be inserted.

        Returns:
            torch.nn.Module: Backbone features with inserted dilations.
        """
        feat = [f for f in features.children()]
        for f in feat:
            if not hasattr(f, 'downsample'):
                for g in f.children():
                    tmp = g
                    if type(tmp) == nn.Conv2d:
                        if tmp.kernel_size != (1, 1):
                            tmp.stride = 1
                            tmp.dilation = (int(dilation_size), int(dilation_size))
                            tmp.padding = (int(dilation_size), int(dilation_size))
            elif f.downsample is None:
                for g in f.children():
                    tmp = g
                    if type(tmp) == nn.Conv2d:
                        if tmp.kernel_size != (1, 1):
                            tmp.stride = 1
                            tmp.dilation = (int(dilation_size), int(dilation_size))
                            tmp.padding = (int(dilation_size), int(dilation_size))
            else:
                if type(f) == models.resnet.BasicBlock:
                    for g in f.children():
                        tmp = g
                        if type(tmp) == nn.Conv2d:
                            if tmp.kernel_size != (1, 1):
                                tmp.stride = 1
                                tmp.dilation = (int(dilation_size), int(dilation_size))
                                tmp.padding = (int(dilation_size), int(dilation_size))
                        elif type(tmp) == nn.Sequential:
                            for k in tmp.children():
                                if type(k) == nn.Conv2d:
                                    k.stride = 1
                else:
                    for g in f.children():
                        tmp = g
                        if type(tmp) == nn.Conv2d:
                            if tmp.kernel_size != (1, 1):
                                tmp.stride = 1
            #                                 tmp.dilation = (int(dilation_size),int(dilation_size))
            #                                 tmp.padding  = (int(dilation_size),int(dilation_size))
                        elif type(tmp) == nn.Sequential:
                            for k in tmp.children():
                                if type(k) == nn.Conv2d:
                                    k.stride = 1
            #                                     k.dilation = (1,1) # (int(dilation_size),int(dilation_size))
            #                                     k.padding  = (0,0) # (int(dilation_size),int(dilation_size))
        return features

    def forward(self, x):
        """
        Forward pass through the Backbone module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        if 'resnet' in self.network_name:
            return self.preset_forward(x)
        else:
            return self._forward(x)

    def _forward(self, x):
        """
        Forward pass through the Backbone module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.backbone(x)

    def preset_forward(self, x, insert_layer=None, return_layer=[1, 2, 3, 4, 5]):
        """
        Forward pass through the Backbone module with preset options.

        Args:
            x (torch.Tensor): Input tensor.
            insert_layer (int, optional): Layer to insert. Defaults to None.
            return_layer (int or list, optional): Layer(s) to return. Defaults to [1, 2, 3, 4, 5].

        Returns:
            OrderedDict: Output tensor(s) with layer names.
        """
        assert (insert_layer is None or return_layer is None or type(return_layer) is list or insert_layer < return_layer)

        if type(return_layer) is int:
            return_layer = [return_layer]
        result = OrderedDict()
        if insert_layer is None or insert_layer == 0:
            x = self.backbone[0](x)
        if 1 in return_layer:
            result['feats_{}_map'.format(1)] = x
            if 1 == max(return_layer):
                return result
        if insert_layer is None or insert_layer <= 1:
            x = self.backbone[1](x)
        if 2 in return_layer:
            result['feats_{}_map'.format(2)] = x
            if 2 == max(return_layer):
                return result
        if insert_layer is None or insert_layer <= 2:
            x = self.backbone[2](x)
        if 3 in return_layer:
            result['feats_{}_map'.format(3)] = x
            if 3 == max(return_layer):
                return result
        if insert_layer is None or insert_layer <= 3:
            x = self.backbone[3](x)
        if 4 in return_layer:
            result['feats_{}_map'.format(4)] = x
            if 4 == max(return_layer):
                return result
        if insert_layer is None or insert_layer <= 4:
            x = self.backbone[4](x)

        result['feats_last_map'] = x

        x = self.avgpool(x)

        x = torch.flatten(x, 1)
        result['feats_pooled'] = x

        return result
