import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
import torch.nn.utils.spectral_norm as spectral_norm
from adain import adaptive_instance_normalization

# Define a basic residual block
class ResBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return nn.ReLU(True)(out)
#        return out


class SPADEResBlk(nn.Module):
    def __init__(self, fin, fout, seg_fin):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        self.conv_0 = spectral_norm(self.conv_0)
        self.conv_1 = spectral_norm(self.conv_1)
        if self.learned_shortcut:
            self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        self.norm_0 = SPADE(fin, seg_fin)
        self.norm_1 = SPADE(fmiddle, seg_fin)
        if self.learned_shortcut:
            self.norm_s = SPADE(fin, seg_fin)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)

class SPADE(nn.Module):
    def __init__(self, cin, seg_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(seg_dim, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.alpha = nn.Conv2d(128, cin,
            kernel_size=3, stride=1, padding=1)
        self.beta = nn.Conv2d(128, cin,
            kernel_size=3, stride=1, padding=1)
            
    @staticmethod
    def PN(x):
        '''
            positional normalization: normalize each positional vector along the channel dimension
        '''
        assert len(x.shape) == 4, 'Only works for 4D(image) tensor'
        x = x - x.mean(dim=1, keepdim=True)
        x_norm = x.norm(dim=1, keepdim=True) + 1e-6
        x = x / x_norm
        return x
        
    def DPN(self, x, s):
        h, w = x.shape[2], x.shape[3]
        s = F.interpolate(s, (h, w), mode='bilinear', align_corners = False)
        s = self.conv(s)
        a = self.alpha(s)
        b  = self.beta(s)
        return x * (1 + a) + b

    def forward(self, x, s):
        x_out = self.DPN(self.PN(x), s)
        return x_out

class ResAdaIN(nn.Module):
    def __init__(self, dim, norm_layer=nn.BatchNorm2d, use_bias=True):
        super(ResAdaIN, self).__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias)
        self.norm1 = norm_layer(dim)
        self.relu = nn.ReLU(True)
        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias)
        self.norm2 = norm_layer(dim)

    def forward(self, x, style):

        residual = x
        x = adaptive_instance_normalization(x,style)
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.pad2(x)
        x = self.conv2(x)
        x = self.norm2(x)
        out = x + residual

        return nn.ReLU(True)(out)
