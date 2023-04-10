from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from data import get_training_set, get_test_set
from util import tensor2img, get_facial_label, get_attention, save_feature_map
from networks import define_G, define_D, GANLoss, VGGLoss, get_scheduler, update_learning_rate
from model import BiSeNet
from torch.autograd import Variable

# Training settings
parser = argparse.ArgumentParser(description='Residual Nets')
parser.add_argument('--dataset', type=str,default='cuhk', help='[cuhk, ar, xmwvts, cuhk_feret, WildSketch]')
parser.add_argument('--output_path', type=str,default='exp1', help='output path')
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--vgg', type=str,default=True, help='use vgg loss?')
parser.add_argument('--cuda', type=str,default=True, help='use cuda?')
parser.add_argument('--visual', action='store_true', help='visualize the result?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')



opt = parser.parse_args()

print(opt)

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

cudnn.benchmark = True

torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
root_path = "data/"
train_set = get_training_set(root_path + opt.dataset)
# test_set = get_test_set(root_path + opt.dataset)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
# testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

device = torch.device("cuda:0" if opt.cuda else "cpu")

n_classes = 19
parsing_net = BiSeNet(n_classes=n_classes).to(device)
parsing_net.load_state_dict(torch.load('face_parsing_bisenet.pth'))
parsing_net.eval()
for param in parsing_net.parameters():
    param.requires_grad = False

net_g_a2b = define_G(opt.input_nc, opt.output_nc, opt.ngf, 'instance', False, 'normal', 0.02, gpu_ids=device)
net_g_b2a = define_G(opt.input_nc, opt.output_nc, opt.ngf, 'instance', False, 'normal', 0.02, gpu_ids=device)
net_d_a2b = define_D(opt.input_nc, opt.ndf, 'basic', gpu_ids=device)
net_d_b2a = define_D(opt.input_nc, opt.ndf, 'basic', gpu_ids=device)

# setup optimizer
criterionL1 = nn.L1Loss().to(device)
criterionMSE = nn.MSELoss().to(device)
criterionGAN = GANLoss().to(device)
if opt.vgg:
    criterionVGG = VGGLoss().to(device)

optimizer_net_g_a2b = optim.Adam(filter(lambda p: p.requires_grad,net_g_a2b.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_net_g_b2a = optim.Adam(filter(lambda p: p.requires_grad,net_g_b2a.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_net_d_a2b = optim.Adam(filter(lambda p: p.requires_grad,net_d_a2b.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_net_d_b2a = optim.Adam(filter(lambda p: p.requires_grad,net_d_b2a.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))

torch.autograd.set_detect_anomaly(True)
for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    # train
    for iteration, batch in enumerate(training_data_loader, 1):
        # a: photo, b: sketch

        # forward
        real_a, real_b = batch[0].to(device), batch[1].to(device)
        ref_a, ref_b = batch[2].to(device), batch[3].to(device)
        
        real_label_a = parsing_net(real_a.detach())[0]
        real_label_b = parsing_net(real_b.detach())[0]

        fake_b = net_g_a2b(real_a,ref_b,real_label_a)
        fake_a = net_g_b2a(real_b,ref_a,real_label_b)

        fake_label_a = parsing_net(fake_a.detach())[0]
        fake_label_b = parsing_net(fake_b.detach())[0]

        rec_b = net_g_a2b(fake_a,ref_b,fake_label_a)
        rec_a = net_g_b2a(fake_b,ref_a,fake_label_b)

        ## train net_d_a2b  
        optimizer_net_d_a2b.zero_grad()
        # train with fake
        pred_fake_a2b = net_d_a2b.forward(fake_b.detach())
        loss_d_a2b_fake = criterionGAN(pred_fake_a2b, False)
        # train with real
        pred_real_a2b = net_d_a2b.forward(real_b)
        loss_d_a2b_real = criterionGAN(pred_real_a2b, True)
        # Combined D loss
        loss_d_a2b = (loss_d_a2b_fake + loss_d_a2b_real) * 0.5
        loss_d_a2b.backward()
        optimizer_net_d_a2b.step()

        ## train net_d_b2a
        optimizer_net_d_b2a.zero_grad()
        # train with fake
        pred_fake_b2a = net_d_b2a.forward(fake_a.detach())
        loss_d_b2a_fake = criterionGAN(pred_fake_b2a, False)
        # train with real
        pred_real_b2a = net_d_b2a.forward(real_a)
        loss_d_b2a_real = criterionGAN(pred_real_b2a, True)
        # Combined D loss
        loss_d_b2a = (loss_d_b2a_fake + loss_d_b2a_real) * 0.5
        loss_d_b2a.backward()
        optimizer_net_d_b2a.step()

        # train net_g
        optimizer_net_g_a2b.zero_grad()
        optimizer_net_g_b2a.zero_grad()
        
        # gan loss
        pred_a2b = net_d_a2b.forward(fake_b)
        loss_g_a2b = criterionGAN(pred_a2b, True)
        # vgg loss
        loss_g_vgg_a2b = criterionVGG(fake_b, real_b)*0.5
        
        # gan loss
        pred_b2a = net_d_b2a.forward(fake_a)
        loss_g_b2a = criterionGAN(pred_b2a, True)
        # vgg loss
        loss_g_vgg_b2a = criterionVGG(fake_a, real_a)*0.5
        
        loss_rec_a = criterionVGG(rec_a, real_a) * 0.5 
        loss_rec_b = criterionVGG(rec_b, real_b) * 0.5

        loss_g = loss_g_a2b + loss_g_b2a + loss_g_vgg_a2b + loss_g_vgg_b2a + loss_rec_a + loss_rec_b
        loss_g.backward()
        
        optimizer_net_g_a2b.step()
        optimizer_net_g_b2a.step()

        print("===> Epoch[{}]({}/{}): real_score: {:.4f}, fake_score: {:.4f}, g_gan_loss: {:.4f}".format(
            epoch, iteration, len(training_data_loader), pred_real_a2b.data.mean().item(), pred_fake_a2b.data.mean().item(), loss_g.item()))
        
    # update_learning_rate(a2b_scheduler, optimizer_net_g_a2b)
    # update_learning_rate(b2a_scheduler, optimizer_net_g_b2a)



    # checkpoint
    if epoch % 2 == 0:
        if not os.path.exists("./checkpoint"):
            os.mkdir("./checkpoint")
        if not os.path.exists(os.path.join("./checkpoint", opt.output_path)):
            os.mkdir(os.path.join("./checkpoint", opt.output_path))
        net_g_a2b_model_out_path = "./checkpoint/{}/netG_a2b_model_epoch_{}.pth".format(opt.output_path, epoch)
        net_g_b2a_model_out_path = "./checkpoint/{}/netG_b2a_model_epoch_{}.pth".format(opt.output_path, epoch)
        net_d_a2b_model_out_path = "./checkpoint/{}/netD_a2b_model_epoch_{}.pth".format(opt.output_path, epoch)
        net_d_b2a_model_out_path = "./checkpoint/{}/netD_b2a_model_epoch_{}.pth".format(opt.output_path, epoch)
        torch.save(net_g_a2b, net_g_a2b_model_out_path)
        torch.save(net_g_b2a, net_g_b2a_model_out_path)
        torch.save(net_d_a2b, net_d_a2b_model_out_path)
        torch.save(net_d_b2a, net_d_b2a_model_out_path)
        print("Checkpoint saved to {}".format("./checkpoint" + opt.output_path))
