from __future__ import print_function
import argparse
import os
import random
import torch
import torchvision.transforms as transforms
from util import is_image_file, load_img, save_img
from model import BiSeNet

# Testing settings
parser = argparse.ArgumentParser(description='Residual Nets')
parser.add_argument('--dataset',type=str,default='cuhk', help='baoji')
parser.add_argument('--output_path', type=str,default='exp1', help='output path')
parser.add_argument('--nepochs', type=int, default=200, help='saved model of which epochs')
parser.add_argument('--cuda', type=str,default=True,help='use cuda')
opt = parser.parse_args()
print(opt)

device = torch.device("cuda:0" if opt.cuda else "cpu")
 
a2b_model_path = "./checkpoint/{}/netG_a2b_model_epoch_{}.pth".format(opt.output_path, opt.nepochs)
b2a_model_path = "./checkpoint/{}/netG_b2a_model_epoch_{}.pth".format(opt.output_path, opt.nepochs)
net_g_a2b = torch.load(a2b_model_path).to(device)
net_g_b2a = torch.load(b2a_model_path).to(device)

n_classes = 19
parsing_net = BiSeNet(n_classes=n_classes).to(device)
parsing_net.load_state_dict(torch.load('face_parsing_bisenet.pth'))
parsing_net.eval()
for param in parsing_net.parameters():
    param.requires_grad = False

a_dir = "data/{}/test/a/".format(opt.dataset)
b_dir = "data/{}/test/b/".format(opt.dataset)

a_image_filenames = [x for x in os.listdir(a_dir) if is_image_file(x)]
b_image_filenames = [x for x in os.listdir(b_dir) if is_image_file(x)]

transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

transform = transforms.Compose(transform_list)

if not os.path.exists('./result'):
    os.mkdir('./result')
if not os.path.exists(os.path.join('./result', opt.output_path)):
    os.mkdir(os.path.join('./result', opt.output_path))
    os.mkdir(os.path.join('./result', opt.output_path,'a2b'))
    os.mkdir(os.path.join('./result', opt.output_path,'b2a'))


for image_name in a_image_filenames:
    img_a,Ha,Wa = load_img(a_dir + image_name)
    img_a = transform(img_a)
    img_a = img_a.unsqueeze(0).to(device)
    a_pfea = parsing_net(img_a.detach())[0]
    m=random.choice([x for x in os.listdir(b_dir) if is_image_file(x)])
    img_b,_,_ = load_img(b_dir + m)
    img_b = transform(img_b)
    img_b = img_b.unsqueeze(0).to(device)
    b_gen = net_g_a2b(img_a,img_b,a_pfea)
    b_gen = b_gen.detach().squeeze(0).cpu()
    save_img(b_gen,Ha,Wa, "./result/{}/{}/{}".format(opt.output_path, 'a2b', image_name))
    del img_a,img_b,b_gen,a_pfea
    torch.cuda.empty_cache()

for image_name in b_image_filenames:
    img_b,Hb,Wb = load_img(b_dir + image_name)
    img_b = transform(img_b)
    img_b = img_b.unsqueeze(0).to(device)
    b_pfea = parsing_net(img_b.detach())[0]
    n=random.choice([x for x in os.listdir(a_dir) if is_image_file(x)])
    img_a,_,_ = load_img(a_dir + n)
    img_a = transform(img_a)
    img_a = img_a.unsqueeze(0).to(device)
    a_gen = net_g_b2a(img_b,img_a,b_pfea)
    a_gen = a_gen.detach().squeeze(0).cpu()
    save_img(a_gen,Hb,Wb, "./result/{}/{}/{}".format(opt.output_path, 'b2a', image_name))
    del img_a,img_b,a_gen,b_pfea
    torch.cuda.empty_cache()
