import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.parameter import Parameter


class parameterNet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(parameterNet, self).__init__()
        # derain parameters
        self.residual_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0),
            nn.PReLU()
        )

        self.background_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0),
            nn.PReLU()
        )
        self.gt_conv = nn.Sequential(
            nn.Conv2d(out_channel*2, out_channel, kernel_size=1, padding=0),
            nn.PReLU()
        )

    def forward(self, n, bilater, res):
        residual = self.residual_conv(res)
        residual = n - residual
        background = self.background_conv(bilater)
        x = torch.cat([residual, background], dim=1)
        gt = self.gt_conv(x)
        return residual, background, gt

class parameterNet_conv3(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(parameterNet_conv3, self).__init__()
        # derain parameters
        self.residual_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            # nn.PReLU()
        )

        self.background_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            # nn.PReLU()
        )
        self.gt_conv = nn.Sequential(
            nn.Conv2d(out_channel*2, out_channel, kernel_size=3, padding=1),
            # nn.PReLU()
        )

    def forward(self, n, bilater, res):
        residual = self.residual_conv(res)
        residual = n - residual
        background = self.background_conv(bilater)
        x = torch.cat([residual, background], dim=1)
        gt = self.gt_conv(x)
        return residual, background, gt

class parameterNet_pure(nn.Module):
    def __init__(self, in_channel, out_channel):
            super(parameterNet_pure, self).__init__()
            # derain parameters

            self.background_conv = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0),
                nn.PReLU()
            )

    def forward(self, n, bilater, res):

        background = self.background_conv(bilater)
        return background


class parameterNet_linearpure(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(parameterNet_linearpure, self).__init__()
            # derain parameters

        self.background_conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0)


    def forward(self, n, bilater, res):

        background = self.background_conv(bilater)
        return background

class parameterNet_linear(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(parameterNet_linear, self).__init__()
        # derain parameters
        self.residual_conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0)

        self.background_conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0)

        self.gt_conv = nn.Conv2d(out_channel*2, out_channel, kernel_size=1, padding=0)


    def forward(self, n, bilater, res):
        residual = self.residual_conv(res)
        residual = n - residual
        background = self.background_conv(bilater)
        x = torch.cat([residual, background], dim=1)
        gt = self.gt_conv(x)
        return residual, background, gt


class parameterNet_mlp(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(parameterNet_mlp, self).__init__()
        # derain parameters
        self.residual_conv = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=1, padding=0),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=1, padding=0),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=1, padding=0),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=1, padding=0),
            nn.PReLU(),
            nn.Conv2d(64, out_channel, kernel_size=1, padding=0),

        )

        self.background_conv = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=1, padding=0),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=1, padding=0),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=1, padding=0),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=1, padding=0),
            nn.PReLU(),
            nn.Conv2d(64, out_channel, kernel_size=1, padding=0),
        )
        self.gt_conv = nn.Sequential(
            nn.Conv2d(out_channel*2, out_channel, kernel_size=1, padding=0),
            nn.PReLU()
        )

    def forward(self, n, bilater, res):
        residual = self.residual_conv(res)
        residual = n - residual
        background = self.background_conv(bilater)
        x = torch.cat([residual, background], dim=1)
        gt = self.gt_conv(x)
        return residual, background, gt
# class residual_block(nn.Module):
#     def __init__(self, feature_dim):
#         super(residual_block, self).__init__()
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
#             nn.BatchNorm2d(feature_dim),
#             nn.PReLU(),
#             nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
#             nn.BatchNorm2d(feature_dim),
#             nn.PReLU()
#         )
#
#     def forward(self, x):
#         return self.double_conv(x)


if __name__ == '__main__':

    net = parameterNet_linear(3*9, 3)
    print('Total number of network parameters is {}'.format(sum(x.numel() for x in net.parameters())))
    for name, param in net.named_parameters():
        # if "derain" in name:
        print(sum(param.size()))