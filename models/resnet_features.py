#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 14:19:24 2023

@author: 
"""


from typing import Callable, List, Optional, Sequence, Tuple, Type, Union

from torch import Tensor
import torch.nn as nn
import torchvision.models as models


# Adapted from:
# https://pytorch.org/vision/main/_modules/torchvision/models/video/resnet.html#r3d_18

class Conv3DSimple(nn.Conv3d):
    def __init__(
        self, in_planes: int, out_planes: int, midplanes: Optional[int] = None, stride: int = 1, padding: int = 1
    ) -> None:

        super().__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(3, 3, 3),
            stride=stride,
            padding=padding,
            bias=False,
        )

    @staticmethod
    def get_downsample_stride(stride: int) -> Tuple[int, int, int]:
        return stride, stride, stride


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        conv_builder: Callable[..., nn.Module],
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        super().__init__()
        self.conv1 = nn.Sequential(
            conv_builder(inplanes, planes, midplanes, stride), nn.BatchNorm3d(planes), nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(conv_builder(planes, planes, midplanes), nn.BatchNorm3d(planes))
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        conv_builder: Callable[..., nn.Module],
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:

        super().__init__()
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        # 1x1x1
        self.conv1 = nn.Sequential(
            nn.Conv3d(inplanes, planes, kernel_size=1, bias=False), nn.BatchNorm3d(planes), nn.ReLU(inplace=True)
        )
        # Second kernel
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes, stride), nn.BatchNorm3d(planes), nn.ReLU(inplace=True)
        )

        # 1x1x1
        self.conv3 = nn.Sequential(
            nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes * self.expansion),
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicStem(nn.Sequential):
    """The default conv-batchnorm-relu stem"""

    def __init__(self) -> None:
        super().__init__(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
   
    
class VideoResNet_features(nn.Module):
    
    '''
    Generic 3DResNet-backbone's generator
    the convolutional layers of 3D ResNet
    the average pooling and final fully convolutional layer is removed
    '''
    
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        conv_makers: Sequence[Type[Union[Conv3DSimple]]],
        layers: List[int],
        stem: Callable[..., nn.Module],
        num_classes: int = 400,
        zero_init_residual: bool = False,
    ) -> None:

        super().__init__()
        self.inplanes = 64

        self.stem = stem()

        self.layer1 = self._make_layer(block, conv_makers[0], 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, conv_makers[1], 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, conv_makers[2], 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, conv_makers[3], 512, layers[3], stride=2)

        #self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        #self.fc = nn.Linear(512 * block.expansion, num_classes)

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[union-attr, arg-type]

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        # Flatten the layer to fc
        # x = x.flatten(1)
        # x = self.fc(x)

        return x

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        conv_builder: Type[Union[Conv3DSimple]],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=ds_stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, conv_builder, stride, downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder))

        return nn.Sequential(*layers)


def video_resnet18_features(pretrained=True) -> VideoResNet_features:
    
    """ Constructs a 3D ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics400
    """
    
    backbone = VideoResNet_features(
        BasicBlock,
        [Conv3DSimple] * 4,
        [2, 2, 2, 2],
        BasicStem,
    )
    
    if pretrained:
        weights = models.video.R3D_18_Weights.DEFAULT # best available weights 
        pretrained_3dresnet18 = models.video.r3d_18(weights=weights)
        weights_dict = pretrained_3dresnet18.state_dict()
        weights_dict.pop('fc.weight')
        weights_dict.pop('fc.bias')
        backbone.load_state_dict(weights_dict, strict=False)
        
    return backbone





