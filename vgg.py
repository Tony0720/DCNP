import torch
import torch.nn as nn
import torchvision 
 

class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(1):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(1, 6):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(6, 11):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(11, 20):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(20, 29):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        c11 = self.slice1(X)
        c21 = self.slice2(c11)
        c31 = self.slice3(c21)
        c41 = self.slice4(c31)
        c51 = self.slice5(c41)
        out = [c11, c21, c31, c41, c51]
        return out