import torch
from torch import nn
import math


# class VAP(nn.Module):
#     def __init__(self, n_segment, feature_dim, num_class, dropout_ratio, groups=16):
#         super(VAP, self).__init__()
#         VAP_level = int(math.log(n_segment, 2))
#         print("=> Using {}-level VAP".format(VAP_level))
#         self.n_segment = n_segment
#         self.VAP_level = VAP_level
#         self.inchannel = 2048
#         total_timescale = 0
#         for i in range(VAP_level):
#             timescale = 2 ** i
#             total_timescale += timescale
#             setattr(self, "VAP_{}".format(timescale),
#                     nn.MaxPool3d((n_segment // timescale, 1, 1), 1, 0, (timescale, 1, 1)))
#             # setattr(self, "VAP_{}".format(timescale),
#             #         nn.Sequential(
#             #             nn.Conv1d(self.inchannel, self.inchannel, kernel_size=n_segment // timescale, stride=1, padding=0, dilation=timescale),
#             #             nn.BatchNorm1d(self.inchannel),
#             #             nn.ReLU(inplace=True)
#             #         ))
#
#         self.GAP = nn.AdaptiveAvgPool1d(1)
#         # self.GAP = nn.AdaptiveMaxPool1d(1)
#         self.TES = nn.Sequential(
#             nn.Linear(total_timescale, total_timescale, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(total_timescale, total_timescale, bias=False),
#         )
#         self.softmax = nn.Softmax(dim=1)
#         self.dropout = nn.Dropout(p=dropout_ratio)
#         self.pred = nn.Linear(feature_dim, num_class)
#
#         # fc init
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight.data, 0, 0.001)
#                 if hasattr(m.bias, 'data'):
#                     nn.init.constant_(m.bias.data, 0)
#             # elif isinstance(m, nn.BatchNorm1d):
#             #         nn.init.constant_(m.weight, 1)
#             #         nn.init.constant_(m.bias, 0)
#
#     def forward(self, x):
#         _, d = x.size()
#         x = x.view(-1, self.n_segment, d, 1, 1).permute(0, 2, 1, 3, 4)
#         x = torch.cat(tuple([getattr(self, "VAP_{}".format(2 ** i))(x) for i in range(self.VAP_level)]), 2).squeeze(3).\
#             squeeze(3).permute(0, 2, 1)
#         # x = torch.cat(tuple([getattr(self, "VAP_{}".format(2 ** i))(x) for i in range(self.VAP_level)]), 2).permute(0, 2, 1)
#         w = self.GAP(x).squeeze(2)
#         w = self.softmax(self.TES(w))
#         x = x * w.unsqueeze(2)
#         x = x.sum(dim=1)
#         # x = x.mean(dim=1)
#         x = self.dropout(x)
#         x = self.pred(x.view(-1, d))
#         return x


class VAP(nn.Module):
    def __init__(self, n_segment, feature_dim, num_class, dropout_ratio, groups=16):
        super(VAP, self).__init__()
        VAP_level = int(math.log(n_segment, 2))
        print("=> Using {}-level VAP".format(VAP_level))
        self.n_segment = n_segment
        self.VAP_level = VAP_level
        self.inchannel = 2048
        total_timescale = 0
        for i in range(VAP_level):
            timescale = 2 ** i
            total_timescale += timescale
            # setattr(self, "VAP_{}".format(timescale),
            #         nn.MaxPool3d((n_segment // timescale, 1, 1), 1, 0, (timescale, 1, 1)))
            setattr(self, "VAP_{}".format(timescale),
                    nn.MaxPool1d(n_segment // timescale, 1, 0, timescale))
            # setattr(self, "VAP_{}".format(timescale),
            #         nn.Sequential(
            #             nn.Conv3d(self.inchannel, self.inchannel, kernel_size=(n_segment // timescale, 1, 1), stride=1, padding=0, dilation=(timescale, 1, 1), bias=False),
            #             nn.BatchNorm3d(self.inchannel),
            #             nn.ReLU(inplace=True)
            #         ))
        self.GAP = nn.AdaptiveMaxPool1d(1)
        # self.TES = nn.Sequential(
        #     nn.Linear(total_timescale, total_timescale * 4, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(total_timescale * 4, total_timescale, bias=False),
        # )
        self.TES = nn.Sequential(
            nn.Linear(self.VAP_level, self.VAP_level * 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.VAP_level * 4, self.VAP_level, bias=False),
            nn.Sigmoid()
        )
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.fc = nn.Linear(feature_dim, num_class)
        # self.GAP3D = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.GAP1D = nn.AdaptiveAvgPool1d(1)

        # fc init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.001)
                if hasattr(m.bias, 'data'):
                    nn.init.constant_(m.bias.data, 0)
            # elif isinstance(m, nn.BatchNorm3d):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)

    def forward(self, x):
        _, d = x.size()
        x = x.view(-1, self.n_segment, d).permute(0, 2, 1)
        if self.VAP_level == 3:
            x1, x2, x3 = tuple([getattr(self, "VAP_{}".format(2 ** i))(x) for i in range(self.VAP_level)])
            x1 = self.GAP1D(x1)
            x2 = self.GAP1D(x2)
            x3 = self.GAP1D(x3)
            x = torch.cat((x1, x2, x3), dim=2).permute(0, 2, 1)
        elif self.VAP_level == 4:
            x1, x2, x3, x4 = tuple([getattr(self, "VAP_{}".format(2 ** i))(x) for i in range(self.VAP_level)])
            x1 = self.GAP1D(x1)
            x2 = self.GAP1D(x2)
            x3 = self.GAP1D(x3)
            x4 = self.GAP1D(x4)
            x = torch.cat((x1, x2, x3, x4), dim=2).permute(0, 2, 1)
        # x = torch.cat(tuple([getattr(self, "VAP_{}".format(2 ** i))(x) for i in range(self.VAP_level)]), 2).squeeze(3).\
        #     squeeze(3).permute(0, 2, 1)
        w = self.GAP(x).squeeze(2)
        w = self.softmax(self.TES(w))
        x = x * w.unsqueeze(2)
        x = x.sum(dim=1)
        # x = x.mean(dim=1)
        x = self.dropout(x)
        x = x.view(-1, d)
        x = self.fc(x.view(-1, d))
        return x