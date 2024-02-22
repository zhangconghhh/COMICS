import torch, pdb
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

"""
Channel Attention and Spaitial Attention from    
Woo, S., Park, J., Lee, J.Y., & Kweon, I. CBAM: Convolutional Block Attention Module. ECCV2018.
"""

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


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
        

class SRMConv2d_simple(nn.Module):    
    def __init__(self, inc=3, learnable=False):
        super(SRMConv2d_simple, self).__init__()
        self.truc = nn.Hardtanh(-3, 3)
        kernel = self._build_kernel(inc)  # (3,3,5,5)
        self.kernel = nn.Parameter(data=kernel, requires_grad=learnable)

    def forward(self, x):
        '''
        x: imgs (Batch, H, W, 3)
        '''
        out = F.conv2d(x, self.kernel, stride=1, padding=2)
        out = self.truc(out)
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
        # filter3：hor 2rd
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
        filters = np.repeat(filters, inc, axis=1)
        filters = torch.FloatTensor(filters)    # (3,3,5,5)
        return filters


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class SRMPixelAttention(nn.Module):
    def __init__(self, in_channels):
        super(SRMPixelAttention, self).__init__()
        self.srm = SRMConv2d_simple()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),)     
        self.pa = SpatialAttention()
        self.pa1 = SpatialAttention()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 2, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        x_srm = self.srm(x)
        fea = self.conv(x_srm) 
        att_map = self.pa(fea)
        fea1 = self.conv1(fea) 
        att_map1 = self.pa(fea1)    
        return att_map, att_map1


class SRMConv2d_Separate(nn.Module):
    def __init__(self, inc, outc, learnable=False):
        super(SRMConv2d_Separate, self).__init__()
        self.inc = inc
        self.truc = nn.Hardtanh(-3, 3)
        kernel = self._build_kernel(inc)  # (3,3,5,5)
        self.kernel = nn.Parameter(data=kernel, requires_grad=learnable) #[384,1,5,5]
        self.out_conv = nn.Sequential(
            nn.Conv2d(3*inc, outc, 1, 1, 0, 1, 1, bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True)
        )
        self.ca = ChannelAttention(outc)
        for ly in self.out_conv.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)

    def forward(self, x):
        '''
        x: imgs (Batch, H, W, 3)
        '''
        out = F.conv2d(x, self.kernel, stride=1, padding=2, groups=self.inc)
        out = self.truc(out)
        out = self.out_conv(out)
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
        filters = np.tile(filters, (inc,1,1,1))
        filters = torch.FloatTensor(filters)    
        return filters
    

class SRMFeature_mvss(nn.Module):
    
    def __init__(self, erb_dim = 256, in_dim = 512, out_dim = 256):
        super(SRMFeature_mvss, self).__init__()    
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

    def forward(self, features):
        output = {}
        c3, c4, c5, c6, c7 = features['p3'], features['p4'], features['p5'], features['p6'], features['p7']            
        srm3 = self.srm_conv1(c3)
        output['p3'] = self.erb_db_1(torch.cat([srm3, c3], dim=1))
        output['p4'] = self.erb_db_2(torch.cat([F.interpolate(srm3, size=(c4.shape[-2], c4.shape[-1]), mode='bilinear', align_corners=True), c4], dim=1))
        srm4 = self.srm_conv1(c4)
        output['p5'] = self.erb_db_3(torch.cat([F.interpolate(srm4, size=(c5.shape[-2], c5.shape[-1]), mode='bilinear', align_corners=True), c5], dim=1))
        srm5 = self.srm_conv1(c5)
        output['p6'] = self.erb_db_4(torch.cat([F.interpolate(srm5, size=(c6.shape[-2], c6.shape[-1]), mode='bilinear', align_corners=True), c6], dim=1))
        srm6 = self.srm_conv1(c6)
        output['p7'] = self.erb_db_5(torch.cat([F.interpolate(srm6, size=(c7.shape[-2], c7.shape[-1]), mode='bilinear', align_corners=True), c7], dim=1))               
        return output



