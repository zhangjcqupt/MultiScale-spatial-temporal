import torch.nn as nn
# from ops.EAB import *
import math
# from tools.download_from_url import download_from_url
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F
import os
from ops.Attention import *

__all__ = ['pyconvresnet50',
           ]

# try:
#     from torch.hub import _get_torch_home
#     torch_cache_home = _get_torch_home()
# except ImportError:
#     torch_cache_home = os.path.expanduser(
#         os.getenv('TORCH_HOME', os.path.join(
#             os.getenv('XDG_CACHE_HOME', '~/.cache'), 'torch')))
# default_cache_path = os.path.join(torch_cache_home, 'pretrained')

model_urls = {
    'pyconvresnet50':'/home/zhangjian/.cache/torch/checkpoints/pyconvresnet50.pth',
    # 'pyconvresnet50': 'https://drive.google.com/uc?export=download&id=128iMzBnHQSPNehgb8nUF5cJyKBIB7do5',
    'pyconvresnet101': 'https://drive.google.com/uc?export=download&id=1fn0eKdtGG7HA30O5SJ1XrmGR_FsQxTb1',
    'pyconvresnet152': 'https://drive.google.com/uc?export=download&id=1zR6HOTaHB0t15n6Nh12adX86AhBMo46m',
}


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# class PyConv4(nn.Module):
#
#     def __init__(self, inplans, planes, pyconv_kernels=[3, 5, 7, 9], stride=1, pyconv_groups=[1, 4, 8, 16]):
#         super(PyConv4, self).__init__()
#         self.num_segment = 16
#         self.conv2_1 = conv(inplans, planes//4, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0]//2,
#                             stride=stride, groups=pyconv_groups[0])
#         self.conv2_2 = conv(inplans, planes//4, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1]//2,
#                             stride=stride, groups=pyconv_groups[1])
#         self.conv2_3 = conv(inplans, planes//4, kernel_size=pyconv_kernels[2], padding=pyconv_kernels[2]//2,
#                             stride=stride, groups=pyconv_groups[2])
#         self.conv2_4 = conv(inplans, planes//4, kernel_size=pyconv_kernels[3], padding=pyconv_kernels[3]//2,
#                             stride=stride, groups=pyconv_groups[3])
#         self.conv11d = nn.Conv3d(inplans // 4, inplans // 4, kernel_size=(1, 1, 1),
#                                  padding=(0, 0, 0), bias=False, groups=inplans // 4)
#         self.conv13d = nn.Conv3d(inplans // 4, inplans // 4, kernel_size=(3, 1, 1),
#                                  padding=(1, 0, 0), bias=False, groups=inplans // 4)
#         self.conv15d = nn.Conv3d(inplans // 4, inplans // 4, kernel_size=(5, 1, 1),
#                                  padding=(2, 0, 0), bias=False, groups=inplans // 4)
#         self.conv17d = nn.Conv3d(inplans // 4, inplans // 4, kernel_size=(7, 1, 1),
#                                  padding=(3, 0, 0), bias=False, groups=inplans // 4)
#         self.weight_init()
#
#         # self.max_pool = nn.MaxPool2d(kernel_size=(3, 3), stride=stride,
#         #                              padding=(1, 1))
#         # self.max_project = nn.Conv2d(inplans, inplans, kernel_size=(1, 1), stride=(1, 1), dilation=1,
#         #                              padding=(0, 0), bias=False)
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.bn = nn.BatchNorm2d(inplans)
#         self.sigmoid = nn.Sigmoid()
#         self.relu = nn.ReLU(inplace=True)
#
#     def ms_groupconv1d(self, x):
#         nt, c, h, w = x.size()
#         n_batch = nt // self.num_segment
#         x_mix = x.view(n_batch, self.num_segment, c, h, w).permute(0, 2, 1, 3, 4).contiguous()  # x => [N, C, T, H, W]
#         x1, x3, x5, x7 = x_mix.split([c // 4, c // 4, c // 4, c // 4], dim=1)
#
#         x1 = self.conv11d(x1).permute(0, 2, 1, 3, 4).contiguous().view(-1, c // 4, h, w)
#         x3 = self.conv13d(x3).permute(0, 2, 1, 3, 4).contiguous().view(-1, c // 4, h, w)
#         x5 = self.conv15d(x5).permute(0, 2, 1, 3, 4).contiguous().view(-1, c // 4, h, w)
#         x7 = self.conv17d(x7).permute(0, 2, 1, 3, 4).contiguous().view(-1, c // 4, h, w)
#
#         x_mix = torch.cat((x1, x3, x5, x7), dim=1).view(nt, c, h, w)
#         y = self.avg_pool(x_mix)
#         y = self.sigmoid(y)
#         x = y.expand_as(x) * x + x
#
#         return x
#
#     def weight_init(self):
#         planes = self.conv11d.in_channels
#         fold = planes // 8 # div = 4
#
#         weight1 = torch.zeros(planes, 1, 1, 1, 1)  # [channels, group_iner_channels, T, H, W] [1]
#         weight1[:, 0, 0] = 1.0
#         self.conv11d.weight = nn.Parameter(weight1)
#
#         # diff 1357 = shift + stride 0 2 4 6
#         weight3 = torch.zeros(planes, 1, 3, 1, 1)  # [channels, group_iner_channels, T, H, W] [010]:1/2 [100]1/4 [110]1/4
#         weight3[:fold, 0, 0] = 1.0
#         weight3[fold: fold * 2, 0, 2] = 1.0
#         weight3[fold * 2:, 0, 1] = 1.0
#         self.conv13d.weight = nn.Parameter(weight3)
#
#         weight5 = torch.zeros(planes, 1, 5, 1, 1)  # [channels, group_iner_channels, T, H, W]
#         weight5[:fold, 0, :2] = 1.0  # [11000]
#         weight5[fold: fold * 2, 0, 3:] = 1.0  # [00011]
#         weight5[fold * 2:, 0, 2] = 1.0 # [00100]
#         self.conv15d.weight = nn.Parameter(weight5)
#
#         weight7 = torch.zeros(planes, 1, 7, 1, 1)  # [channels, group_iner_channels, T, H, W]
#         weight7[:fold, 0, :3] = 1.0  # [1110000]
#         weight7[fold: fold * 2, 0, 4:] = 1.0  # [0000111]
#         weight7[fold * 2:, 0, 3] = 1.0 # [0001000]
#         self.conv17d.weight = nn.Parameter(weight7)
#
#         # self.conv11d.weight.requires_grad = True
#         # self.conv13d.weight.requires_grad = True
#         # self.conv15d.weight.requires_grad = True
#         # self.conv17d.weight.requires_grad = True
#
#     def forward(self, x):
#         # x_max = self.max_pool(x)
#         # x_max = self.max_project(x_max)
#         # x_max = self.bn(x_max)
#
#         x_temporal = self.ms_groupconv1d(x)
#         # x = x_temporal
#
#         x1 = self.conv2_1(x_temporal)
#         x2 = self.conv2_2(x_temporal)
#         x3 = self.conv2_3(x_temporal)
#         x4 = self.conv2_4(x_temporal)
#         x = torch.cat((x1, x2, x3, x4), dim=1)
#
#         # x = x
#
#         return x
#
#         # return torch.cat((self.conv2_1(x), self.conv2_2(x), self.conv2_3(x), self.conv2_4(x)), dim=1)
#
#
# class PyConv3(nn.Module):
#
#     def __init__(self, inplans, planes,  pyconv_kernels=[3, 5, 7], stride=1, pyconv_groups=[1, 4, 8]):
#         super(PyConv3, self).__init__()
#         self.num_segment = 16
#         self.conv2_1 = conv(inplans, planes // 4, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0] // 2,
#                             stride=stride, groups=pyconv_groups[0])
#         self.conv2_2 = conv(inplans, planes // 4, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1] // 2,
#                             stride=stride, groups=pyconv_groups[1])
#         self.conv2_3 = conv(inplans, planes // 2, kernel_size=pyconv_kernels[2], padding=pyconv_kernels[2] // 2,
#                             stride=stride, groups=pyconv_groups[2])
#         self.conv11d = nn.Conv3d(inplans // 4, inplans // 4, kernel_size=(1, 1, 1),
#                                  padding=(0, 0, 0), bias=False, groups=inplans // 4)
#         self.conv13d = nn.Conv3d(inplans // 4, inplans // 4, kernel_size=(3, 1, 1),
#                                  padding=(1, 0, 0), bias=False, groups=inplans // 4)
#         self.conv15d = nn.Conv3d(inplans // 2, inplans // 2, kernel_size=(5, 1, 1),
#                                  padding=(2, 0, 0), bias=False, groups=inplans // 2)
#         self.weight_init()
#
#         # self.max_pool = nn.MaxPool2d(kernel_size=(3, 3), stride=stride,
#         #                              padding=(1, 1))
#         # self.max_project = nn.Conv2d(inplans, inplans, kernel_size=(1, 1), stride=(1, 1), dilation=1,
#         #                              padding=(0, 0), bias=False)
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.bn = nn.BatchNorm2d(inplans)
#         self.sigmoid = nn.Sigmoid()
#         self.relu = nn.ReLU(inplace=True)
#
#     def ms_groupconv1d(self, x):
#         nt, c, h, w = x.size()
#         n_batch = nt // self.num_segment
#         x_mix = x.view(n_batch, self.num_segment, c, h, w).permute(0, 2, 1, 3, 4).contiguous()  # x => [N, C, T, H, W]
#         x1, x3, x5 = x_mix.split([c // 4, c // 4, c // 2], dim=1)
#
#         x1 = self.conv11d(x1).permute(0, 2, 1, 3, 4).contiguous().view(-1, c // 4, h, w)
#         x3 = self.conv13d(x3).permute(0, 2, 1, 3, 4).contiguous().view(-1, c // 4, h, w)
#         x5 = self.conv15d(x5).permute(0, 2, 1, 3, 4).contiguous().view(-1, c // 2, h, w)
#
#         x_mix = torch.cat((x1, x3, x5), dim=1).view(nt, c, h, w)
#         y = self.avg_pool(x_mix)
#         y = self.sigmoid(y)
#         x = y.expand_as(x) * x + x
#
#         return x
#
#     def weight_init(self):
#         planes = self.conv11d.in_channels
#         fold = planes // 8  # div = 4
#
#         weight1 = torch.zeros(planes, 1, 1, 1, 1)  # [channels, group_iner_channels, T, H, W] [1]
#         weight1[:, 0, 0] = 1.0
#         self.conv11d.weight = nn.Parameter(weight1)
#
#         # diff 1357 = shift + stride 0 2 4 6
#         weight3 = torch.zeros(planes, 1, 3, 1,
#                               1)  # [channels, group_iner_channels, T, H, W] [010]:1/2 [100]1/4 [110]1/4
#         weight3[:fold, 0, 0] = 1.0
#         weight3[fold: fold * 2, 0, 2] = 1.0
#         weight3[fold * 2:, 0, 1] = 1.0
#         self.conv13d.weight = nn.Parameter(weight3)
#
#         weight5 = torch.zeros(planes * 2, 1, 5, 1, 1)  # [channels, group_iner_channels, T, H, W]
#         weight5[:fold, 0, :2] = 1.0  # [11000]
#         weight5[fold: fold * 2, 0, 3:] = 1.0  # [00011]
#         weight5[fold * 2:, 0, 2] = 1.0  # [00100]
#         self.conv15d.weight = nn.Parameter(weight5)
#
#         # self.conv11d.weight.requires_grad = True
#         # self.conv13d.weight.requires_grad = True
#         # self.conv15d.weight.requires_grad = True
#
#     def forward(self, x):
#         # x_max = self.max_pool(x)
#         # x_max = self.max_project(x_max)
#         # x_max = self.bn(x_max)
#
#         x_temporal = self.ms_groupconv1d(x)
#         x = x_temporal
#
#         x1 = self.conv2_1(x)
#         x2 = self.conv2_2(x)
#         x3 = self.conv2_3(x)
#         x = torch.cat((x1, x2, x3), dim=1)
#
#         x = x
#
#         return x
#
#         # return torch.cat((self.conv2_1(x), self.conv2_2(x), self.conv2_3(x)), dim=1)
#
#
# class PyConv2(nn.Module):
#
#     def __init__(self, inplans, planes,pyconv_kernels=[3, 5], stride=1, pyconv_groups=[1, 4]):
#         super(PyConv2, self).__init__()
#         self.num_segment = 16
#         self.conv2_1 = conv(inplans, planes // 2, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0] // 2,
#                             stride=stride, groups=pyconv_groups[0])
#         self.conv2_2 = conv(inplans, planes // 2, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1] // 2,
#                             stride=stride, groups=pyconv_groups[1])
#         self.conv11d = nn.Conv3d(inplans // 2, inplans // 2, kernel_size=(1, 1, 1),
#                                  padding=(0, 0, 0), bias=False, groups=inplans // 2)
#         self.conv13d = nn.Conv3d(inplans // 2, inplans // 2, kernel_size=(3, 1, 1),
#                                  padding=(1, 0, 0), bias=False, groups=inplans // 2)
#         self.weight_init()
#
#         # self.max_pool = nn.MaxPool2d(kernel_size=(3, 3), stride=stride,
#         #                              padding=(1, 1))
#         # self.max_project = nn.Conv2d(inplans, inplans, kernel_size=(1, 1), stride=(1, 1), dilation=1,
#         #                              padding=(0, 0), bias=False)
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.bn = nn.BatchNorm2d(inplans)
#         self.sigmoid = nn.Sigmoid()
#         self.relu = nn.ReLU(inplace=True)
#
#     def ms_groupconv1d(self, x):
#         nt, c, h, w = x.size()
#         n_batch = nt // self.num_segment
#         x_mix = x.view(n_batch, self.num_segment, c, h, w).permute(0, 2, 1, 3, 4).contiguous()  # x => [N, C, T, H, W]
#         x1, x3 = x_mix.split([c // 2, c // 2], dim=1)
#
#         x1 = self.conv11d(x1).permute(0, 2, 1, 3, 4).contiguous().view(-1, c // 2, h, w)
#         x3 = self.conv13d(x3).permute(0, 2, 1, 3, 4).contiguous().view(-1, c // 2, h, w)
#
#         x_mix = torch.cat((x1, x3), dim=1).view(nt, c, h, w)
#         y = self.avg_pool(x_mix)
#         y = self.sigmoid(y)
#         x = y.expand_as(x) * x + x
#
#         return x
#
#     def weight_init(self):
#         planes = self.conv11d.in_channels
#         fold = planes // 8  # div = 4
#
#         weight1 = torch.zeros(planes, 1, 1, 1, 1)  # [channels, group_iner_channels, T, H, W] [1]
#         weight1[:, 0, 0] = 1.0
#         self.conv11d.weight = nn.Parameter(weight1)
#
#         # diff 1357 = shift + stride 0 2 4 6
#         weight3 = torch.zeros(planes, 1, 3, 1,
#                               1)  # [channels, group_iner_channels, T, H, W] [010]:1/2 [100]1/4 [110]1/4
#         weight3[:fold, 0, 0] = 1.0
#         weight3[fold: fold * 2, 0, 2] = 1.0
#         weight3[fold * 2:, 0, 1] = 1.0
#         self.conv13d.weight = nn.Parameter(weight3)
#
#         # self.conv11d.weight.requires_grad = True
#         # self.conv13d.weight.requires_grad = True
#
#
#     def forward(self, x):
#         # x_max = self.max_pool(x)
#         # x_max = self.max_project(x_max)
#         # x_max = self.bn(x_max)
#
#         x_temporal = self.ms_groupconv1d(x)
#         x = x_temporal
#
#         x1 = self.conv2_1(x)
#         x2 = self.conv2_2(x)
#         x = torch.cat((x1, x2), dim=1)
#
#         return x
#
#         # return torch.cat((self.conv2_1(x), self.conv2_2(x)), dim=1)
#
#
# def get_pyconv(inplans, planes, pyconv_kernels, stride=1, pyconv_groups=[1]):
#     if len(pyconv_kernels) == 1:
#         return conv(inplans, planes, kernel_size=pyconv_kernels[0], stride=stride, groups=pyconv_groups[0])
#     elif len(pyconv_kernels) == 2:
#         return PyConv2(inplans, planes, pyconv_kernels=pyconv_kernels, stride=stride, pyconv_groups=pyconv_groups)
#     elif len(pyconv_kernels) == 3:
#         return PyConv3(inplans, planes, pyconv_kernels=pyconv_kernels, stride=stride, pyconv_groups=pyconv_groups)
#     elif len(pyconv_kernels) == 4:
#         return PyConv4(inplans, planes, pyconv_kernels=pyconv_kernels, stride=stride, pyconv_groups=pyconv_groups)

class PyConv4(nn.Module):

    def __init__(self, inplans, planes, pyconv_kernels=[3, 5, 7, 9], stride=1, pyconv_groups=[1, 4, 8, 16]):
        super(PyConv4, self).__init__()
        self.num_segments = 16
        self.conv2_1 = conv(inplans, planes//4, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0]//2,
                            stride=stride, groups=pyconv_groups[0])
        self.conv2_2 = conv(inplans, planes//4, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1]//2,
                            stride=stride, groups=pyconv_groups[1])
        self.conv2_3 = conv(inplans, planes//4, kernel_size=pyconv_kernels[2], padding=pyconv_kernels[2]//2,
                            stride=stride, groups=pyconv_groups[2])
        self.conv2_4 = conv(inplans, planes//4, kernel_size=pyconv_kernels[3], padding=pyconv_kernels[3]//2,
                            stride=stride, groups=pyconv_groups[3])
        self.channel_att = SEweightModule(inplans // 4)
        self.split_channel = planes // 4
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]

        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x3 = self.conv2_3(x)
        x4 = self.conv2_4(x)
        feats = torch.cat((x1, x2, x3, x4), dim=1)
        feats_weight = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])

        x1_channel = self.channel_att(x1)
        x2_channel = self.channel_att(x2)
        x3_channel = self.channel_att(x3)
        x4_channel = self.channel_att(x4)

        x_channel = torch.cat((x1_channel, x2_channel, x3_channel, x4_channel), dim=1)
        attention_vectors = x_channel.view(batch_size, 4, self.split_channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats_weight * attention_vectors
        # for i in range(4):
        #     x_se_weight_fp = feats_weight[:, i, :, :]
        #     if i == 0:
        #         out = x_se_weight_fp
        #     else:
        #         out = torch.cat((x_se_weight_fp, out), 1)
        out = feats_weight.view(batch_size, 4 * self.split_channel, feats_weight.shape[3],
                                feats_weight.shape[4]) #+ feats
        # out = feats_weight.reshape(batch_size, 4 * self.split_channel, feats_weight.shape[3], feats_weight.shape[4])

        return out
        # return torch.cat((self.conv2_1(x), self.conv2_2(x), self.conv2_3(x), self.conv2_4(x)), dim=1)


class PyConv3(nn.Module):

    def __init__(self, inplans, planes,  pyconv_kernels=[3, 5, 7], stride=1, pyconv_groups=[1, 4, 8]):
        super(PyConv3, self).__init__()
        self.num_segments = 16
        self.conv2_1 = conv(inplans, planes // 4, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0] // 2,
                            stride=stride, groups=pyconv_groups[0])
        self.conv2_2 = conv(inplans, planes // 4, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1] // 2,
                            stride=stride, groups=pyconv_groups[1])
        self.conv2_3 = conv(inplans, planes // 2, kernel_size=pyconv_kernels[2], padding=pyconv_kernels[2] // 2,
                            stride=stride, groups=pyconv_groups[2])
        self.channel_att = SEweightModule(inplans // 4)
        self.split_channel = planes // 4
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]

        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x3 = self.conv2_3(x)
        # x = torch.cat((x1, x2, x3), dim=1)

        x3, x4 = x3.split([self.split_channel, self.split_channel], dim=1)

        feats = torch.cat((x1, x2, x3, x4), dim=1)
        feats_weight = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])

        x1_channel = self.channel_att(x1)
        x2_channel = self.channel_att(x2)
        x3_channel = self.channel_att(x3)
        x4_channel = self.channel_att(x4)

        x_channel = torch.cat((x1_channel, x2_channel, x3_channel, x4_channel), dim=1)
        attention_vectors = x_channel.view(batch_size, 4, self.split_channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats_weight * attention_vectors

        out = feats_weight.view(batch_size, 4 * self.split_channel, feats_weight.shape[3],
                                feats_weight.shape[4]) #+ feats
        # out = feats_weight.reshape(batch_size, 4 * self.split_channel, feats_weight.shape[3], feats_weight.shape[4]) + feats

        return out

        # return torch.cat((self.conv2_1(x), self.conv2_2(x), self.conv2_3(x)), dim=1)


class PyConv2(nn.Module):

    def __init__(self, inplans, planes,pyconv_kernels=[3, 5], stride=1, pyconv_groups=[1, 4]):
        super(PyConv2, self).__init__()
        self.num_segments = 16
        self.conv2_1 = conv(inplans, planes // 2, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0] // 2,
                            stride=stride, groups=pyconv_groups[0])
        self.conv2_2 = conv(inplans, planes // 2, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1] // 2,
                            stride=stride, groups=pyconv_groups[1])
        self.channel_att = SEweightModule(inplans // 2)
        self.split_channel = planes // 2
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]

        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        feats = torch.cat((x1, x2), dim=1)
        feats_weight = feats.view(batch_size, 2, self.split_channel, feats.shape[2], feats.shape[3])

        x1_channel = self.channel_att(x1)
        x2_channel = self.channel_att(x2)

        x_channel = torch.cat((x1_channel, x2_channel), dim=1)
        attention_vectors = x_channel.view(batch_size, 2, self.split_channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats_weight * attention_vectors
        # for i in range(2):
        #     x_se_weight_fp = feats_weight[:, i, :, :]
        #     if i == 0:
        #         out = x_se_weight_fp
        #     else:
        #         out = torch.cat((x_se_weight_fp, out), 1)
        out = feats_weight.view(batch_size, 2 * self.split_channel, feats_weight.shape[3],
                                feats_weight.shape[4]) #+ feats
        # out = feats_weight.reshape(batch_size, 2 * self.split_channel, feats_weight.shape[3], feats_weight.shape[4]) + feats

        return out

        # return torch.cat((self.conv2_1(x), self.conv2_2(x)), dim=1)


def get_pyconv(inplans, planes, pyconv_kernels, stride=1, pyconv_groups=[1]):
    if len(pyconv_kernels) == 1:
        return conv(inplans, planes, kernel_size=pyconv_kernels[0], stride=stride, groups=pyconv_groups[0])
    elif len(pyconv_kernels) == 2:
        return PyConv2(inplans, planes, pyconv_kernels=pyconv_kernels, stride=stride, pyconv_groups=pyconv_groups)
    elif len(pyconv_kernels) == 3:
        return PyConv3(inplans, planes, pyconv_kernels=pyconv_kernels, stride=stride, pyconv_groups=pyconv_groups)
    elif len(pyconv_kernels) == 4:
        return PyConv4(inplans, planes, pyconv_kernels=pyconv_kernels, stride=stride, pyconv_groups=pyconv_groups)

# class PyConv4(nn.Module):
#
#     def __init__(self, inplans, planes, pyconv_kernels=[3, 5, 7, 9], stride=1, pyconv_groups=[1, 4, 8, 16]):
#         super(PyConv4, self).__init__()
#         self.conv2_1 = conv(inplans, planes//4, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0]//2,
#                             stride=stride, groups=pyconv_groups[0])
#         self.conv2_2 = conv(inplans, planes//4, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1]//2,
#                             stride=stride, groups=pyconv_groups[1])
#         self.conv2_3 = conv(inplans, planes//4, kernel_size=pyconv_kernels[2], padding=pyconv_kernels[2]//2,
#                             stride=stride, groups=pyconv_groups[2])
#         self.conv2_4 = conv(inplans, planes//4, kernel_size=pyconv_kernels[3], padding=pyconv_kernels[3]//2,
#                             stride=stride, groups=pyconv_groups[3])
#
#     def forward(self, x):
#         return torch.cat((self.conv2_1(x), self.conv2_2(x), self.conv2_3(x), self.conv2_4(x)), dim=1)
#
#
# class PyConv3(nn.Module):
#
#     def __init__(self, inplans, planes,  pyconv_kernels=[3, 5, 7], stride=1, pyconv_groups=[1, 4, 8]):
#         super(PyConv3, self).__init__()
#         self.conv2_1 = conv(inplans, planes // 4, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0] // 2,
#                             stride=stride, groups=pyconv_groups[0])
#         self.conv2_2 = conv(inplans, planes // 4, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1] // 2,
#                             stride=stride, groups=pyconv_groups[1])
#         self.conv2_3 = conv(inplans, planes // 2, kernel_size=pyconv_kernels[2], padding=pyconv_kernels[2] // 2,
#                             stride=stride, groups=pyconv_groups[2])
#
#     def forward(self, x):
#         return torch.cat((self.conv2_1(x), self.conv2_2(x), self.conv2_3(x)), dim=1)
#
#
# class PyConv2(nn.Module):
#
#     def __init__(self, inplans, planes,pyconv_kernels=[3, 5], stride=1, pyconv_groups=[1, 4]):
#         super(PyConv2, self).__init__()
#         self.conv2_1 = conv(inplans, planes // 2, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0] // 2,
#                             stride=stride, groups=pyconv_groups[0])
#         self.conv2_2 = conv(inplans, planes // 2, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1] // 2,
#                             stride=stride, groups=pyconv_groups[1])
#
#     def forward(self, x):
#         return torch.cat((self.conv2_1(x), self.conv2_2(x)), dim=1)
#
#
# def get_pyconv(inplans, planes, pyconv_kernels, stride=1, pyconv_groups=[1]):
#     if len(pyconv_kernels) == 1:
#         return conv(inplans, planes, kernel_size=pyconv_kernels[0], stride=stride, groups=pyconv_groups[0])
#     elif len(pyconv_kernels) == 2:
#         return PyConv2(inplans, planes, pyconv_kernels=pyconv_kernels, stride=stride, pyconv_groups=pyconv_groups)
#     elif len(pyconv_kernels) == 3:
#         return PyConv3(inplans, planes, pyconv_kernels=pyconv_kernels, stride=stride, pyconv_groups=pyconv_groups)
#     elif len(pyconv_kernels) == 4:
#         return PyConv4(inplans, planes, pyconv_kernels=pyconv_kernels, stride=stride, pyconv_groups=pyconv_groups)


class PyConvBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, num_segments, stride=1, downsample=None, pyconv_groups=1, pyconv_kernels=1):
        super(PyConvBlock, self).__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = get_pyconv(planes, planes, pyconv_kernels=pyconv_kernels, stride=stride,
                                pyconv_groups=pyconv_groups)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.num_segments = num_segments

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# from ops.SOI_transformer import *

class PyConvResNet(nn.Module):

    def __init__(self, block, layers, num_segments, num_classes=1000, zero_init_residual=False, ):
        super(PyConvResNet, self).__init__()

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.num_segments = num_segments

        self.layer1 = self._make_layer(block, 64, layers[0], num_segments=num_segments, stride=2, pyconv_kernels=[3, 5, 7, 9], pyconv_groups=[1, 4, 8, 16])
        self.layer2 = self._make_layer(block, 128, layers[1], num_segments=num_segments, stride=2, pyconv_kernels=[3, 5, 7], pyconv_groups=[1, 4, 8])
        self.layer3 = self._make_layer(block, 256, layers[2], num_segments=num_segments, stride=2, pyconv_kernels=[3, 5], pyconv_groups=[1, 4])
        self.layer4 = self._make_layer(block, 512, layers[3], num_segments=num_segments, stride=2, pyconv_kernels=[3], pyconv_groups=[1])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # elif isinstance(m, nn.BatchNorm1d):
            #     nn.init.constant_(m.weight, 0.001)
            #     nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.BatchNorm3d):
            #     nn.init.constant_(m.weight, 0.001)
            #     nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, PyConvBlock):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, num_segments, stride=1, pyconv_kernels=[3], pyconv_groups=[1]):
        downsample = None
        if stride != 1 and self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=stride, padding=1),
                conv1x1(self.inplanes, planes * block.expansion),
                nn.BatchNorm2d(planes * block.expansion),
            )
        elif self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion),
                nn.BatchNorm2d(planes * block.expansion),
            )
        elif stride != 1:
            downsample = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)

        layers = []
        layers.append(block(self.inplanes, planes, num_segments, stride=stride, downsample=downsample,
                            pyconv_kernels=pyconv_kernels, pyconv_groups=pyconv_groups))
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, num_segments, pyconv_kernels=pyconv_kernels, pyconv_groups=pyconv_groups))

        return nn.Sequential(*layers)

    def get_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x


def pyconvresnet50(pretrained=True, num_segments = 16, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = PyConvResNet(PyConvBlock, [3, 4, 6, 3], num_segments=num_segments)
    if pretrained:
        # os.makedirs(default_cache_path, exist_ok=True)
        model.load_state_dict(torch.load('/raid/zhangj/source/pyconvresnet50.pth'), strict=False)
    return model

#
# def pyconvresnet101(pretrained=True, num_segments = 8,progress=True, **kwargs):
#     """Constructs a ResNet-101 model.
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     model = PyConvResNet(PyConvBlock, [3, 4, 23, 3], num_segments=num_segments)
#     if pretrained:
#         os.makedirs(default_cache_path, exist_ok=True)
#         model.load_state_dict(torch.load(download_from_url(model_urls['pyconvresnet101'],
#                                                            root=default_cache_path)))
#     return model





