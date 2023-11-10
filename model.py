
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import math
import torch.utils.model_zoo as model_zoo
from torchvision.ops import DeformConv2d
import torchvision.models as torch_models


model_urls = {
    'resnet18': torch_models.ResNet18_Weights.DEFAULT.url,
    'resnet34': torch_models.ResNet34_Weights.DEFAULT.url,
    'resnet50': torch_models.ResNet50_Weights.DEFAULT.url,
    'resnet101': torch_models.ResNet101_Weights.DEFAULT.url,
    'resnet152': torch_models.ResNet152_Weights.DEFAULT.url,
}

# Initialising offset Conv2d
def constant_init(module, constant, bias=0):
    nn.init.constant_(module.weight, constant)
    if hasattr(module, 'bias'):
        nn.init.constant_(module.bias, bias)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

# BasicBlock consists of two convolutional layers only used in resnet18,resnet34
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dcn=False):
        super(BasicBlock, self).__init__()
        self.with_dcn = dcn
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        if not self.with_dcn:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        else:
            # deformable_groups = dcn.get('deformable_groups', 1)
            deformable_groups = 1
            offset_channels = 18
            self.conv2_offset = nn.Conv2d(planes, deformable_groups * offset_channels, kernel_size=3, padding=1)
            self.conv2 = DeformConv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if not self.with_dcn:
            out = self.conv2(out)
        else:
            offset = self.conv2_offset(out)
            out = self.conv2(out, offset)

        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# Consists of three Convolutional Layers, used as building blocks in resnet50 & higher
class Bottleneck(nn.Module):
    expansion = 4
    # Set dcn to True to replace 3*3 convolution with 3*3 deformable convolution
    def __init__(self, inplanes, planes, stride=1, downsample=None, dcn=False):
        super(Bottleneck, self).__init__()
        self.with_dcn = dcn
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # Replace 3*3 normal conv2d with deformconv2d if dcn=True
        if not self.with_dcn:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        else:
            # deformable_groups = dcn.get('deformable_groups', 1)
            deformable_groups = 1
            offset_channels = 18
            # Replacing with deformable convolution
            self.conv2_offset = nn.Conv2d(planes, deformable_groups * offset_channels, stride=stride, kernel_size=3, padding=1)
            self.conv2 = DeformConv2d(planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if not self.with_dcn:
            out = self.conv2(out)
        else:
            offset = self.conv2_offset(out)
            out = self.conv2(out, offset)

        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Resnet(nn.Module):
    """
        dcn: list of int to specify how many layers in each conv block to replace with deformable convolution
        unfreeze_conv: list of int to specific how many layers of deformable or convolution layers to unfreeze in each conv block
    """
    def __init__(self, block, layers, num_classes, dcn=[0,0,0,0],unfreeze_conv=[0,0,0,0],unfreeze_offset=True,unfreeze_fc=True):
        super(Resnet, self).__init__()
        self.inplanes = 64
        self.dcn = dcn
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.out_channels = []
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Building ResNet conv block
        self.layer1 = self._make_layer(block, 64, layers[0],dcn=dcn[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,dcn=dcn[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dcn=dcn[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dcn=dcn[3])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_fc = nn.Linear(self.out_channels[2] * 2, num_classes)

        # Initilialising offset weights as 0
        if self.dcn is not None:
            for m in self.modules():
                if isinstance(m, Bottleneck) or isinstance(m, BasicBlock):
                    if hasattr(m, 'conv2_offset'):
                        constant_init(m.conv2_offset, 0)

        self.out_shape = [self.out_channels[0] * 2,
                          self.out_channels[1] * 2,
                          self.out_channels[2] * 2]


        print("backbone output channel: C3 {}, C4 {}, C5 {}".format(self.out_channels[0] * 2, self.out_channels[1] * 2, self.out_channels[2] * 2))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
       
        # Freezing layer for fine tune training
        self.freeze_all_layers()
        # Unfreezing offsets
        if unfreeze_offset:
            self.unfreeze_offset()
        #Unfreezing fully connected layer
        if unfreeze_fc:
            self.unfreeze_fc()

        # Unfreezing selected 3*3 conv layer
        self.unfreeze_conv(unfreeze_conv)
    
    def freeze_all_layers(self):
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze_dcn(self):
        for module in self.modules():
            if isinstance(module, DeformConv2d):
                for param in module.parameters():
                    param.requires_grad = True
    
    def unfreeze_conv(self, last_layer):
        # Get the corresponding block
        for idx,num_layer in enumerate(last_layer):
            block = getattr(self, f'layer{idx+1}')
            if num_layer>0:
                # Freeze the 3x3 conv in the specified layers
                for layer in block[-num_layer:]:
                    layer.conv2.requires_grad = True           

    def unfreeze_offset(self):
        for name, param in self.named_parameters():
            if 'offset' in name:
                param.requires_grad = True
    
    def unfreeze_fc(self):
        for module in self.modules():
            if isinstance(module,nn.Linear):
                for param in module.parameters():
                    param.requires_grad = True

    def _make_layer(self, block, planes, blocks, stride=1, dcn=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        
        # True to set 3x3 conv to dcn
        dcns = ([False]*(blocks-dcn)) + ([True]*dcn)
        layers = [block(self.inplanes, planes, stride, downsample, dcn=dcns[0])]
        self.inplanes = planes * block.expansion
        self.out_channels.append(self.inplanes)

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dcn=dcns[i]))

        return nn.Sequential(*layers)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x4 = self.avgpool(x4)
        x = torch.flatten(x4, 1)
        x = self.output_fc(x)

        y = torch.softmax(x, dim=1)
        
        return y


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Resnet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='.'), strict=False)
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Resnet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir='.'), strict=False)
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Resnet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='.'), strict=False)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Resnet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='.'), strict=False)
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Resnet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir='.'), strict=False)
    return model


def resnet(pretrained=False, **kwargs):
    version = str(kwargs.pop('version'))
    if version == '18':
        return resnet18(pretrained, **kwargs)
    if version == '34':
        return resnet34(pretrained, **kwargs)
    if version == '50':
        return resnet50(pretrained, **kwargs)
    if version == '101':
        return resnet101(pretrained, **kwargs)
    if version == '152':
        return resnet152(pretrained, **kwargs)