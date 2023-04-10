import torch
import torch.nn as nn

class DBFM(nn.Module):
    def __init__(self, nf):
        super(DBFM, self).__init__()
        self.conv_a = nn.Conv2d(nf, nf, 1, 1, 0, bias=True)
        self.conv_gate_a = nn.Conv2d(nf, nf, 1, 1, 0, bias=True)
        self.conv_b = nn.Conv2d(nf, nf, 1, 1, 0, bias=True)
        self.conv_gate_b = nn.Conv2d(nf, nf, 1, 1, 0, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            nn.BatchNorm2d(nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            nn.BatchNorm2d(nf),
            nn.ReLU(inplace=True)
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(nf, int(nf//4), kernel_size=1, stride=1, padding=0),
#            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(nf//4),nf, kernel_size=1, stride=1, padding=0),
#            nn.BatchNorm2d(channels),
        )

    def forward(self, a, b):
        a = self.conv_a(a)
        g_a = self.conv_gate_a(a)
        g_a = self.sigmoid(g_a)
        b = self.conv_b(b)
        g_b = self.conv_gate_b(b)
        g_b = self.sigmoid(g_b)
        x = self.global_att(a+b)
        wei = self.sigmoid(x)
        a_gff = wei*(1+g_a)*a+(1-wei)*(1+g_b)*b
        out = self.conv(a_gff)

        return out
        
class AFF(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
#            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
#            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)


        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo