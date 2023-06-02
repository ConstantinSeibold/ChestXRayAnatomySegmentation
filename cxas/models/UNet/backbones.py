import torch
import pdb,os
import torch.nn as nn
import torchvision
import torchvision.models as m
from collections import OrderedDict
BatchNorm = nn.BatchNorm2d
import torch.nn.functional as F

class backbone(nn.Module):
    def __init__(self, network_name):
        super(backbone, self).__init__()
        network_name = network_name.split('_')[1].lower()
        self.network_name = network_name
        self.backbone = self.get_backbone(network_name)        
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1,1))

    def get_backbone(self,network_name):
        if network_name == 'vgg16':
            full_net = getattr(m, network_name)()
            features = list(full_net.features)[:30]
            
            net =  nn.Sequential(*features)

            self.classifier = nn.Sequential(*list(full_net.classifier)[:5])

        elif 'resnet' in network_name:
            full_net = getattr(m, network_name)()

            features = [
                            ('layer0',torch.nn.Sequential(*[full_net.conv1,
                            full_net.bn1,
                            full_net.relu,
                            full_net.maxpool])),
                            ('layer1',full_net.layer1),
                            ('layer2',full_net.layer2),
                            ('layer3',full_net.layer3),
                            ('layer4',full_net.layer4),
                        ]

            self.inplanes =  full_net.inplanes

            net =  nn.Sequential(OrderedDict(features))
        else:
            raise '{} not implemented as BACKBONE Network'.format(network_name)

        return net

    
    def _make_layer(self, block, inplanes, planes, blocks, stride=1, dilation=1, new_level=True, residual=True):
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
        feat = [f for f in features.children()]
        for f in feat:
            if not hasattr(f,'downsample'):
                for g in f.children():
                    tmp = g
                    if type(tmp) == nn.Conv2d:
                        if tmp.kernel_size != (1,1) :
                            tmp.stride   = 1
                            tmp.dilation = (int(dilation_size),int(dilation_size))
                            tmp.padding  = (int(dilation_size),int(dilation_size))
            elif f.downsample is None:
                for g in f.children():
                    tmp = g
                    if type(tmp) == nn.Conv2d:
                        if tmp.kernel_size != (1,1) :
                            tmp.stride   = 1
                            tmp.dilation = (int(dilation_size),int(dilation_size))
                            tmp.padding  = (int(dilation_size),int(dilation_size))
            else:
                if type(f) == torchvision.models.resnet.BasicBlock:
                    for g in f.children():
                        tmp = g
                        if type(tmp) == nn.Conv2d:
                            if tmp.kernel_size != (1,1) :
                                tmp.stride   = 1
                                tmp.dilation = (int(dilation_size),int(dilation_size))
                                tmp.padding  = (int(dilation_size),int(dilation_size))
                        elif type(tmp) == nn.Sequential:
                            for k in tmp.children():
                                if type(k) == nn.Conv2d:
                                    k.stride= 1
                else:
                    for g in f.children():
                        tmp = g
                        if type(tmp) == nn.Conv2d:
                            if tmp.kernel_size != (1,1) :
                                tmp.stride   = 1
                                # tmp.dilation = (int(dilation_size),int(dilation_size))
                                # tmp.padding  = (int(dilation_size),int(dilation_size))
                        elif type(tmp) == nn.Sequential:
                            for k in tmp.children():
                                if type(k) == nn.Conv2d:
                                    k.stride= 1
                                    # k.dilation = (1,1) # (int(dilation_size),int(dilation_size))
                                    # k.padding  = (0,0) # (int(dilation_size),int(dilation_size))
        return features

    def forward(self,x):
        if 'resnet' in self.network_name:
            return self.preset_forward(x)
        else:
            return self._forward(x)

    def _forward(self, x ):
        return self.backbone(x)

    def preset_forward(self, x,  insert_layer = None, return_layer = [1,2,3,4,5]):
        assert (insert_layer is None or return_layer is None or type(return_layer) is list or insert_layer < return_layer )

        if type(return_layer) is int:
            return_layer = [return_layer]
        result = OrderedDict()
        if insert_layer is None or insert_layer == 0:
            x = self.backbone[0](x)
        if  1 in return_layer:
            result['feats_{}_map'.format(1)] = x
            if 1 == max(return_layer):
                return result
        if insert_layer is None or insert_layer <= 1:
            x = self.backbone[1](x)
        if  2 in return_layer:
            result['feats_{}_map'.format(2)] = x
            if 2 == max(return_layer):
                return result
        if insert_layer is None or insert_layer <= 2:
            x = self.backbone[2](x)
        if  3 in return_layer:
            result['feats_{}_map'.format(3)] = x
            if 3 == max(return_layer):
                return result
        if insert_layer is None or insert_layer <= 3:
            x = self.backbone[3](x)
        if  4 in return_layer:
            result['feats_{}_map'.format(4)] = x
            if 4 == max(return_layer):
                return result
        if insert_layer is None or insert_layer <= 4:
            x = self.backbone[4](x)

        result['feats_last_map'] = x

        x = self.avgpool(x)

        x = torch.flatten(x, 1)
        result['feats_pooled']  = x

        return result
