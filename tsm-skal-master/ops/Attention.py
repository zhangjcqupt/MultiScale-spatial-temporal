import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter


class Spatail_attention_3(nn.Module):
    def __init__(self, inplanes, num_segments):
        super(Spatail_attention_3, self).__init__()
        self.inplane = inplanes
        self.num_segment = num_segments
        self.sigmoid = nn.Sigmoid()

        #spatial temporal excitation
        self.st_conv = nn.Conv3d(1, 1, kernel_size=(3, 3, 3),
                                         stride=(1, 1, 1), bias=False, padding=(1, 1, 1))

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.num_segment
        x_p1 = x.view(n_batch, self.num_segment, c, h, w).transpose(2, 1).contiguous()
        x_p1 = x_p1.sum(1, keepdim=True)
        x_p1 = self.st_conv(x_p1)
        x_p1 = x_p1.transpose(2, 1).contiguous().view(nt, 1, h, w)
        x_p1 = self.sigmoid(x_p1)
        x_p1 = x * x_p1 #+ x

        return x_p1


class Spatail_attention_9(nn.Module):
    def __init__(self, inplanes, num_segments):
        super(Spatail_attention_9, self).__init__()
        self.inplane = inplanes
        self.num_segment = num_segments
        self.sigmoid = nn.Sigmoid()

        #spatial temporal excitation
        self.st_conv = nn.Conv3d(1, 1, kernel_size=(7, 9, 9),
                                         stride=(1, 1, 1), bias=False, padding=(3, 4, 4))

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.num_segment
        x_p1 = x.view(n_batch, self.num_segment, c, h, w).transpose(2, 1).contiguous()
        x_p1 = x_p1.mean(1, keepdim=True)
        x_p1 = self.st_conv(x_p1)
        x_p1 = x_p1.transpose(2, 1).contiguous().view(nt, 1, h, w)
        x_p1 = self.sigmoid(x_p1)
        x_p1 = x * x_p1 #+ x

        return x_p1

class Channel_attention(nn.Module):
    def __init__(self, inplanes, num_segments):
        super(Channel_attention, self).__init__()
        self.inplanes = inplanes
        self.reduced_channels = self.inplanes // 16
        self.num_segment = num_segments
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

        self.relu = nn.ReLU(inplace=True)

        #Channel excitation
        self.Channel_conv1 = nn.Conv2d(self.inplanes, self.reduced_channels, kernel_size=(1, 1), stride=(1, 1),
                                           bias=False, padding=(0, 0))
        # self.bn1 = nn.BatchNorm2d(self.reduced_channels)
        self.Channel_conv2 = nn.Conv1d(self.reduced_channels, self.reduced_channels, kernel_size=3, stride=1,
                                         bias=False, padding=1, groups=1)
        # self.bn2 = nn.BatchNorm1d(self.reduced_channels)
        # self.Channel_conv2 = nn.Conv3d(self.reduced_channels, self.reduced_channels, kernel_size=(3, 1, 1),
        #                                stride=(1, 1, 1),
        #                                bias=False, padding=(1, 0, 0), groups=1)
        # self.bn2 = nn.BatchNorm3d(self.reduced_channels)
        self.Channel_conv3 = nn.Conv2d(self.reduced_channels, self.inplanes, kernel_size=(1, 1), stride=(1, 1),
                                          bias=False, padding=(0, 0))
        # self.bn3 = nn.BatchNorm2d(self.inplanes)

    def forward(self, x):
        x_channel = self.avg_pool(x)
        x_channel = self.Channel_conv1(x_channel)
        # x_channel = self.bn1(x_channel)
        nt, c, h, w = x_channel.size()
        n_batch = nt // self.num_segment
        x_channel = x_channel.view(n_batch, self.num_segment, c, 1, 1).squeeze(-1).squeeze(-1).transpose(2, 1).contiguous()
        # x_channel = x_channel.view(n_batch, self.num_segment, c, 1, 1).transpose(2, 1).contiguous()
        x_channel = self.Channel_conv2(x_channel)
        # x_channel = self.bn2(x_channel)
        x_channel = self.relu(x_channel)
        x_channel = x_channel.transpose(2, 1).contiguous().view(-1, c, 1, 1)
        x_channel = self.Channel_conv3(x_channel)
        # x_channel = self.bn3(x_channel)
        x_channel = self.sigmoid(x_channel)
        # x = x * x_channel #+ x

        return x_channel

class Channel_Att(nn.Module):
    def __init__(self, channels):
        super(Channel_Att, self).__init__()
        self.channels = channels
        self.bn = nn.BatchNorm2d(self.channels, affine=True)

    def forward(self, x):
        residual = x

        x = self.bn(x)
        weight_bn = self.bn.weight.data.abs() / torch.sum(self.bn.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = torch.sigmoid(x) * residual

        return x


class Spatial_Att(nn.Module):
    def __init__(self, channels):
        super(Spatial_Att, self).__init__()
        self.channels = channels
        self.bn = nn.BatchNorm2d(self.channels, affine=True)

    def forward(self, x):
        b, c, h, w = x.size()

        residual = x

        x = x.permute(0, 2, 3, 1).reshape(b, h*w, 1, c).contiguous()
        x = self.bn(x)

        weight_bn = self.bn.weight.data.abs() / torch.sum(self.bn.weight.data.abs())

        x = x.permute(0, 3, 2, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.reshape(b, c, h, w).contiguous()

        x = torch.sigmoid(x) * residual

        return x


class SEweightModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEweightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)

        return weight

class AFF(nn.Module):
    def __init__(self, channels, reduction=4):
        super(AFF, self).__init__()
        self.channels = channels
        self.inter_channels = self.channels // reduction

        self.local_att = nn.Sequential(
            nn.Conv2d(self.channels, self.inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inter_channels, self.channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.channels)
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.channels, self.inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inter_channels, self.channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.channels)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        weight = self.sigmoid(xlg)

        xo = 2 * x * weight + 2 * residual * (1 - weight)

        return xo

class SKA(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SKA, self).__init__()
        self.channels = channels
        self.inter_channels = self.channels // reduction
        self.channels_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.channels, self.inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inter_channels, self.channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.channels)
        )
        self.softmax = nn.Softmax(dim=1)
        # self.fc3 = nn.Conv2d(channels * 2, channels, kernel_size=1, padding=0)

        # self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        batch_size = x1.shape[0]
        feats = torch.cat((x1, x2), dim=1)

        feats_weight = feats.view(batch_size, 2, self.channels, feats.shape[2], feats.shape[3])
        x1_channel = self.channels_att(x1)
        x2_channel = self.channels_att(x2)

        x_channel = torch.cat((x1_channel, x2_channel), dim=1)
        attention_vectors = x_channel.view(batch_size, 2, self.channels, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats_weight * attention_vectors
        out = feats_weight.sum(1).view(batch_size, self.channels, feats_weight.shape[3], feats_weight.shape[4])

        return out


class AFFchange(nn.Module):
    def __init__(self, channels, reduction=4):
        super(AFFchange, self).__init__()
        self.channels = channels
        self.inter_channels = self.channels // reduction

        # self.local_att1 = nn.Sequential(
        #     nn.Conv2d(self.channels, self.inter_channels, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(self.inter_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.inter_channels, self.channels, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(self.channels)
        # )

        # self.local_att2 = nn.Sequential(
        #     nn.Conv2d(self.channels, self.inter_channels, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(self.inter_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.inter_channels, self.channels, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(self.channels)
        # )

        self.global_att1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.channels, self.inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inter_channels, self.channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.channels)
        )
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.channels, self.inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inter_channels, self.channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.channels)
        )

        self.sigmoid = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()

    def forward(self, x1, x2):
        batch_size = x1.shape[0]
        xa = x1 + x2
        # xl = self.local_att1(xa)
        xg = self.global_att1(xa)
        xlg = xg
        weight = self.sigmoid(xlg)

        xo = x1 * weight + x2 * (1 - weight)

        # xl2 = self.local_att2(xo)
        xg2 = self.global_att2(xo)
        xlg2 = xg2
        weight2 = self.sigmoid2(xlg2)

        xo2 = x1 * weight2 + x2 * (1 - weight2)
        xo2 = xo2.view(batch_size, self.channels, x1.size(2), x1.size(3))

        return xo2

        # batch_size = x1.shape[0]
        # xg = self.global_att1(x1)
        # xl = self.local_att1(x2)
        # xlg = xl + xg
        # weight = self.sigmoid(xlg)
        #
        # xo = x1 * weight + x2 * (1 - weight)
        #
        # xl2 = self.local_att2(xo)
        # xg2 = self.global_att2(xo)
        # xlg2 = xl2 + xg2
        # weight2 = self.sigmoid2(xlg2)
        #
        # xo2 = x1 * weight2 + x2 * (1 - weight2)
        # xo2 = xo2.view(batch_size, self.channels, x1.size(2), x1.size(3))
        #
        # return xo2