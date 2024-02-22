import torch, pdb
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision
import numpy as np


class ERB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ERB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, relu=True):
        x = self.conv1(x)
        res = self.conv2(x)
        res = self.bn(res)
        res = self.relu(res)
        res = self.conv3(res)
        if relu:
            return self.relu(x + res)
        else:
            return x+res


class SRMConv2d_Separate(nn.Module):
    
    def __init__(self, inc, outc, learnable=False):
        super(SRMConv2d_Separate, self).__init__()
        self.inc = inc
        self.truc = nn.Hardtanh(-3, 3)
        kernel = self._build_kernel(inc)  # (3,3,5,5)
        self.kernel = nn.Parameter(data=kernel, requires_grad=learnable)
        self.out_conv = nn.Sequential(
            nn.Conv2d(3*inc, outc, 1, 1, 0, 1, 1, bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True))
        for ly in self.out_conv.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)

    def forward(self, x): 
        '''
        x: imgs (Batch, H, W, 3) [8, 256, 108, 128]
        '''
        out = F.conv2d(x, self.kernel, stride=1, padding=2, groups=self.inc) # [8, 768, 108, 128]， self.kernel [768, 1, 5, 5]
        out = self.truc(out) # [8, 768, 108, 128]
        out = self.out_conv(out) # [8, 256, 108, 128]

        return out

    def _build_kernel(self, inc):
        # filter1: KB
        filter1 = [[0, 0, 0, 0, 0],
                   [0, -1, 2, -1, 0],
                   [0, 2, -4, 2, 0],
                   [0, -1, 2, -1, 0],
                   [0, 0, 0, 0, 0]]
        # filter2：KV
        filter2 = [[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]]
        # # filter3：hor 2rd
        filter3 = [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 1, -2, 1, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]]
        filter1 = np.asarray(filter1, dtype=float) / 4.
        filter2 = np.asarray(filter2, dtype=float) / 12.
        filter3 = np.asarray(filter3, dtype=float) / 2.
        # statck the filters
        filters = [[filter1], [filter2], [filter3]]
        filters = np.array(filters) 
        filters = np.repeat(filters, inc, axis=0) # (768, 1, 5, 5)
        filters = torch.FloatTensor(filters)    # (3,3,5,5)
        
        return filters


class SRMFeature_FCOS(nn.Module):
    
    def __init__(self, erb_dim = 256, in_dim = 256, out_dim = 256):
        super(SRMFeature_FCOS, self).__init__()
        self.downsample_2 = nn.Upsample(scale_factor=0.5, mode="bilinear", align_corners=True)
        self.erb_dim = erb_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.erb_db_1 = ERB(self.in_dim, self.erb_dim)
        self.erb_db_2 = ERB(self.in_dim, self.erb_dim)
        self.erb_db_3 = ERB(self.in_dim, self.erb_dim)
        self.erb_db_4 = ERB(self.in_dim, self.erb_dim)
        self.erb_db_5= ERB(self.in_dim, self.erb_dim)
        self.srm_conv1 = SRMConv2d_Separate(256, 256)
        self.srm_conv2 = SRMConv2d_Separate(256, 256)
        self.srm_conv3 = SRMConv2d_Separate(256, 256)
        self.srm_conv4 = SRMConv2d_Separate(256, 256)
        self.srm_conv5 = SRMConv2d_Separate(256, 256)        
        self.down1 = nn.AvgPool2d(3, stride=2, padding=1)
        self.down2 = nn.AvgPool2d(3, stride=2, padding=1)
        self.down3 = nn.AvgPool2d(3, stride=2, padding=1)
        self.down4 = nn.AvgPool2d(3, stride=2, padding=1)

    def forward(self, features):
        c1, c2, c3, c4, c5 = features[0], features[1], features[2], features[3], features[4]            
        res1 = self.erb_db_1(self.srm_conv1(c1))              
        res2 = self.erb_db_2(self.srm_conv2(c2))
        res3 = self.erb_db_3(self.srm_conv3(c3))
        res4 = self.erb_db_4(self.srm_conv4(c4))    
        res5 = self.erb_db_5(self.srm_conv5(c5))    
        features = []
        features.append(res1)
        features.append(res2)
        features.append(res3)
        features.append(res4)
        features.append(res5)

        return features


    