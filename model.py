
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import math
import torch.utils.model_zoo as model_zoo
from torchvision.ops import DeformConv2d


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class DropBlock2D(nn.Module):
    r"""Randomly zeroes 2D spatial blocks of the input tensor.
    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.
    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop
    Shape:
        - Input: `(N, C, H, W)`
        - Output: `(N, C, H, W)`
    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890
    """

    def __init__(self, drop_prob, block_size):
        super(DropBlock2D, self).__init__()

        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        # shape: (bsize, channels, height, width)

        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma(x)

            # sample mask
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()

            # place mask on input device
            mask = mask.to(x.device)

            # compute block mask
            block_mask = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask[:, None, :, :]

            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size ** 2)


class LinearScheduler(nn.Module):
    def __init__(self, dropblock, start_value, stop_value, nr_steps):
        super(LinearScheduler, self).__init__()
        self.dropblock = dropblock
        self.i = 0
        self.drop_values = np.linspace(start=start_value, stop=stop_value, num=int(nr_steps))

    def forward(self, x):
        return self.dropblock(x)

    def step(self):
        if self.i < len(self.drop_values):
            self.dropblock.drop_prob = self.drop_values[self.i]

        self.i += 1

class CBAM(nn.Module):

    def __init__(self, n_channels_in, reduction_ratio, kernel_size):
        super(CBAM, self).__init__()
        self.n_channels_in = n_channels_in
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size

        self.channel_attention = ChannelAttention(n_channels_in, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, f):
        chan_att = self.channel_attention(f)
        fp = chan_att * f
        spat_att = self.spatial_attention(fp)
        fpp = spat_att * fp
        return fpp


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size):
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size

        assert kernel_size % 2 == 1, "Odd kernel size required"
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size,
                              padding=int((kernel_size - 1) / 2))

    def forward(self, x):
        max_pool = self.agg_channel(x, "max")
        avg_pool = self.agg_channel(x, "avg")
        pool = torch.cat([max_pool, avg_pool], dim=1)
        conv = self.conv(pool)
        conv = conv.repeat(1, x.size()[1], 1, 1)
        att = torch.sigmoid(conv)
        return att

    def agg_channel(self, x, pool="max"):
        b, c, h, w = x.size()
        x = x.view(b, c, h * w)
        x = x.permute(0, 2, 1)
        if pool == "max":
            x = F.max_pool1d(x, int(c))
        elif pool == "avg":
            x = F.avg_pool1d(x, int(c))
        x = x.permute(0, 2, 1)
        x = x.view(b, 1, h, w)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, n_channels_in, reduction_ratio):
        super(ChannelAttention, self).__init__()
        self.n_channels_in = n_channels_in
        self.reduction_ratio = reduction_ratio
        self.middle_layer_size = int(self.n_channels_in / float(self.reduction_ratio))

        self.bottleneck = nn.Sequential(
            nn.Linear(self.n_channels_in, self.middle_layer_size),
            nn.ReLU(),
            nn.Linear(self.middle_layer_size, self.n_channels_in)
        )

    def forward(self, x):
        kernel = (x.size()[2], x.size()[3])
        avg_pool = F.avg_pool2d(x, kernel)
        max_pool = F.max_pool2d(x, kernel)

        avg_pool = avg_pool.view(avg_pool.size()[0], -1)
        max_pool = max_pool.view(max_pool.size()[0], -1)

        avg_pool_bck = self.bottleneck(avg_pool)
        max_pool_bck = self.bottleneck(max_pool)

        pool_sum = avg_pool_bck + max_pool_bck

        sig_pool = torch.sigmoid(pool_sum)
        sig_pool = sig_pool.unsqueeze(2).unsqueeze(3)

        out = sig_pool.repeat(1, 1, kernel[0], kernel[1])
        return out


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

    def __init__(self, inplanes, planes, stride=1, downsample=None, cbam=False, dcn=False):
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
        self.use_cbam = cbam
        if self.use_cbam:
            self.cbam = CBAM(n_channels_in=self.expansion * planes, reduction_ratio=1, kernel_size=3)
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

        if self.use_cbam:
            out = self.cbam(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# Consists of three Convolutional Layers, used as building blocks in resnet50 & higher
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, cbam=False, dcn=False):
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
            self.conv2_offset = nn.Conv2d(planes, deformable_groups * offset_channels, stride=stride, kernel_size=3, padding=1)
            self.conv2 = DeformConv2d(planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.use_cbam = cbam
        if self.use_cbam:
            self.cbam = CBAM(n_channels_in=self.expansion * planes, reduction_ratio=1, kernel_size=3)
        

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

        if self.use_cbam:
            out = self.cbam(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# added method to freeze all layers except the offset layers
class Resnet(nn.Module):

    def __init__(self, block, layers, cbam=False, dcn=False,unfreeze_dcn=True,unfreeze_offset=True,unfreeze_fc=True, drop_prob=0):
        super(Resnet, self).__init__()
        self.inplanes = 64
        self.dcn = dcn
        self.cbam = cbam
        self.drop = drop_prob != 0
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.out_channels = []
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        if self.drop:
            self.dropblock = LinearScheduler(
                DropBlock2D(drop_prob=drop_prob, block_size=5),
                start_value=0.,
                stop_value=drop_prob,
                nr_steps=5e3
            )
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, cbam=self.cbam)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, cbam=self.cbam, dcn=dcn)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, cbam=self.cbam, dcn=dcn)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_fc = nn.Linear(self.out_channels[2] * 2, num_classes)

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
        
       

        self.freeze_all_layers()
        if unfreeze_dcn:
            self.unfreeze_dcn()
        if unfreeze_offset:
            self.unfreeze_offset()
        if unfreeze_fc:
            self.unfreeze_fc
    
    def freeze_all_layers(self):
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze_dcn(self):
        for module in self.modules():
            if isinstance(module, DeformConv2d):
                for param in module.parameters():
                    param.requires_grad = True

    def unfreeze_offset(self):
        for name, param in self.named_parameters():
            if 'offset' in name:
                param.requires_grad = True
    
    def unfreeze_fc(self):
        for module in self.modules():
            if isinstance(module,nn.Linear):
                for param in module.parameters():
                    param.requires_grad=True

    def _make_layer(self, block, planes, blocks, stride=1, cbam=False, dcn=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, cbam=cbam, dcn=dcn)]
        self.inplanes = planes * block.expansion
        self.out_channels.append(self.inplanes)
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cbam=cbam, dcn=dcn))

        return nn.Sequential(*layers)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        if self.drop:
            x1 = self.dropblock(self.layer1(x))
            x2 = self.dropblock(self.layer2(x1))
        else:
            x1 = self.layer1(x)
            x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        xx4 = self.avgpool(x4)
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