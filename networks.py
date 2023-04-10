#from typing_extensions import Concatenate
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import math
from modules import ResBlock,SPADEResBlk,ResAdaIN
from vgg import VGG19
from fusion import AFF, DBFM
from torch.autograd import Variable


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


# update learning rate (called once every epoch)
def update_learning_rate(scheduler, optimizer):
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    print('learning rate = %.7f' % lr)


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids='cuda:0'):
    net.to(gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids='cuda:0'):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    net = TransferNet2(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)

    return init_net(net, init_type, init_gain, gpu_ids)


class TransferNet2(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(TransferNet2, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.p_in = Inconv(input_nc, ngf, norm_layer, use_bias)
        self.p_down1 = Down(ngf, ngf * 2, norm_layer, use_bias)
        self.p_down2 = Down(ngf * 2, ngf * 4, norm_layer, use_bias)
        self.p_down3 = Down(ngf * 4, ngf * 8, norm_layer, use_bias)
        self.spade1 = SPADEResBlk(128,128,19)
        self.spade2 = SPADEResBlk(256,256,19)
        self.spade3 = SPADEResBlk(512,512,19)
        
        self.resblock1=nn.Sequential(
            ResAdaIN(ngf * 2),
            ResAdaIN(ngf * 2),
            ResAdaIN(ngf * 2),
            ResAdaIN(ngf * 2),
            ResAdaIN(ngf * 2),
            ResAdaIN(ngf * 2),
            ResAdaIN(ngf * 2),
            ResAdaIN(ngf * 2),
            ResAdaIN(ngf * 2)
        )

        self.resblock2=nn.Sequential(
            ResAdaIN(ngf * 4),
            ResAdaIN(ngf * 4),
            ResAdaIN(ngf * 4),
            ResAdaIN(ngf * 4),
            ResAdaIN(ngf * 4),
            ResAdaIN(ngf * 4),
            ResAdaIN(ngf * 4),
            ResAdaIN(ngf * 4),
            ResAdaIN(ngf * 4)
        )

        self.resblock3=nn.Sequential(
            ResAdaIN(ngf * 8),
            ResAdaIN(ngf * 8),
            ResAdaIN(ngf * 8),
            ResAdaIN(ngf * 8),
            ResAdaIN(ngf * 8),
            ResAdaIN(ngf * 8),
            ResAdaIN(ngf * 8),
            ResAdaIN(ngf * 8),
            ResAdaIN(ngf * 8)
        )
        
        self.up1_p = Up(ngf * 8, ngf * 4, norm_layer, use_bias)
        self.up2_p = Up(ngf * 4, ngf * 2, norm_layer, use_bias)
        self.up3_p = Up(ngf * 2, ngf, norm_layer, use_bias)
        self.out_conv_p = Outconv(ngf, output_nc)

        self.cat1 = AFF(channels=ngf*4)
        self.cat2 = AFF(channels=ngf*2)
        self.fusion1 = DBFM(128)
        self.fusion2 = DBFM(256)
        self.fusion3 = DBFM(512)

    def forward(self, photo, sketch, mask_p):
        mask_p = torch.softmax(mask_p,1)

        fp_64 = self.p_in(photo)
        fs_64 = self.p_in(sketch)
        fp_128 = self.p_down1(fp_64)
        fs_128 = self.p_down1(fs_64)
        fp_256 = self.p_down2(fp_128)
        fs_256 = self.p_down2(fs_128)
        fp_512 = self.p_down3(fp_256)
        fs_512 = self.p_down3(fs_256)
        
        for i in range(9):
            fp_128 = self.resblock1[i](fp_128,fs_128)        
        fp_128_spade = self.spade1(fs_128,mask_p)
        fp_128_ = self.fusion1(fp_128,fp_128_spade)
        
        for i in range(9):
            fp_256 = self.resblock2[i](fp_256,fs_256)        
        fp_256_spade = self.spade2(fs_256,mask_p)
        fp_256_ = self.fusion2(fp_256,fp_256_spade)
        
        for i in range(9):
            fp_512 = self.resblock3[i](fp_512,fs_512)        
        fp_512_spade = self.spade3(fs_512,mask_p)
        fp_512_ = self.fusion3(fp_512,fp_512_spade)

        fp_256__ = self.up1_p(fp_512_)
        fp_256_ = self.cat1(fp_256_,fp_256__)

        fp_128__ = self.up2_p(fp_256_)
        fp_128_ = self.cat2(fp_128_,fp_128__)

        fp_64 = self.up3_p(fp_128_)
        fake_sketch = self.out_conv_p(fp_64)
        
        return fake_sketch


class Inconv(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer, use_bias):
        super(Inconv, self).__init__()
        self.inconv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, out_ch, kernel_size=7, padding=0,
                      bias=use_bias),
            norm_layer(out_ch),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.inconv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer, use_bias):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3,
                      stride=2, padding=1, bias=use_bias),
            norm_layer(out_ch),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.down(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer, use_bias):
        super(Up, self).__init__()
        self.up = nn.Sequential(
             #nn.Upsample(scale_factor=2, mode='nearest'),
             #nn.Conv2d(in_ch, out_ch,
             #          kernel_size=3, stride=1,
             #          padding=1, bias=use_bias),
            nn.ConvTranspose2d(in_ch, out_ch,
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=use_bias),
            norm_layer(out_ch),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Outconv, self).__init__()
        self.outconv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, out_ch, kernel_size=7, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.outconv(x)
        return x


def define_D(input_nc, ndf, netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', init_gain=0.02, gpu_ids='cuda:0'):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'n_layers':
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)

    return init_net(net, init_type, init_gain, gpu_ids)



# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss