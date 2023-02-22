import torch
from torch import nn
import torch.nn.functional as F
# from torchsummary import summary
from .pvtv2 import pvt_v2_b2
import numpy as np
import cv2


class feature_fuse(nn.Module):
    def __init__(self, in_channel=128, out_channel=128):
        super(feature_fuse, self).__init__()
        self.dim = in_channel
        self.out_dim = out_channel
        self.fuseconv = nn.Sequential(nn.Conv2d(2 * self.dim, self.out_dim, 1, 1, 0, bias=False),
                                      nn.BatchNorm2d(self.out_dim),
                                      nn.ReLU(True))
        self.conv = nn.Sequential(nn.Conv2d(self.out_dim, self.out_dim, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(self.out_dim),
                                  nn.ReLU(True))

    def forward(self, Ri, Di):
        assert Ri.ndim == 4
        RDi = torch.cat((Ri, Di), dim=1)
        RDi = self.fuseconv(RDi)
        RDi = self.conv(RDi)
        return RDi
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, n_feat, kernel_size=3, reduction=16,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(self.default_conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def default_conv(self, in_channels, out_channels, kernel_size, bias=True):
        return nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size // 2), bias=bias)

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

class _AtrousSpatialPyramidPoolingModule(nn.Module):
    '''
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    '''

    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=[6, 12, 18]):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()

        # Check if we are using distributed BN and use the nn from encoding.nn
        # library rather than using standard pytorch.nn

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                          nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                nn.ReLU(inplace=True)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True))
        self.edge_conv = nn.Sequential(
            nn.Conv2d(1, reduction_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = F.interpolate(img_features, x_size[2:],
                                     mode='bilinear', align_corners=True)
        out = img_features

        # edge_features = F.interpolate(edge, x_size[2:],
        #                               mode='bilinear', align_corners=True)
        # edge_features = self.edge_conv(edge_features)
        # out = torch.cat((out, edge_features), 1)

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out


class Edge_Module(nn.Module):

    def __init__(self, in_fea=[128, 320, 512], mid_fea=32):
        super(Edge_Module, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_fea[0], mid_fea, 1)
        self.conv4 = nn.Conv2d(in_fea[1], mid_fea, 1)
        self.conv5 = nn.Conv2d(in_fea[2], mid_fea, 1)
        self.conv5_2 = nn.Conv2d(mid_fea, mid_fea, 3, padding=1)
        self.conv5_4 = nn.Conv2d(mid_fea, mid_fea, 3, padding=1)
        self.conv5_5 = nn.Conv2d(mid_fea, mid_fea, 3, padding=1)

        self.classifer = nn.Conv2d(mid_fea * 3, 1, kernel_size=3, padding=1)
        self.rcab = RCAB(mid_fea * 3)

    def forward(self,input, x2, x4, x5):
        _, _, h, w = input.size()
        edge2_fea = self.relu(self.conv2(x2))
        edge2 = self.relu(self.conv5_2(edge2_fea))
        edge4_fea = self.relu(self.conv4(x4))
        edge4 = self.relu(self.conv5_4(edge4_fea))
        edge5_fea = self.relu(self.conv5(x5))
        edge5 = self.relu(self.conv5_5(edge5_fea))

        edge2 = F.interpolate(edge2, size=(h, w), mode='bilinear', align_corners=True)
        edge4 = F.interpolate(edge4, size=(h, w), mode='bilinear', align_corners=True)
        edge5 = F.interpolate(edge5, size=(h, w), mode='bilinear', align_corners=True)

        edge = torch.cat([edge2, edge4, edge5], dim=1)
        edge = self.rcab(edge)
        edge = self.classifer(edge)
        return edge

class Decoder(nn.Module):
    def __init__(self, dim=128):
        super(Decoder, self).__init__()
        self.dim = dim
        self.out_dim = dim
        self.fuse1 = feature_fuse(in_channel=64, out_channel=128)
        self.fuse2 = feature_fuse(in_channel=128, out_channel=128)
        self.fuse3 = feature_fuse(in_channel=320, out_channel=128)
        self.fuse4 = feature_fuse(in_channel=320, out_channel=128)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.up4 = nn.Upsample(scale_factor=4, mode="bilinear")

        self.Conv43 = nn.Sequential(nn.Conv2d(2 * self.out_dim, self.out_dim, 1, 1, 0, bias=False),
                                    nn.BatchNorm2d(self.out_dim),
                                    nn.ReLU(True), nn.Conv2d(self.out_dim, self.out_dim, 3, 1, 1, bias=False),
                                    nn.BatchNorm2d(self.out_dim),
                                    nn.ReLU(True))

        self.Conv432 = nn.Sequential(nn.Conv2d(2 * self.out_dim, self.out_dim, 1, 1, 0, bias=False),
                                     nn.BatchNorm2d(self.out_dim),
                                     nn.ReLU(True), nn.Conv2d(self.out_dim, self.out_dim, 3, 1, 1, bias=False),
                                     nn.BatchNorm2d(self.out_dim),
                                     nn.ReLU(True))
        self.Conv4321 = nn.Sequential(nn.Conv2d(2 * self.out_dim, self.out_dim, 1, 1, 0, bias=False),
                                      nn.BatchNorm2d(self.out_dim),
                                      nn.ReLU(True), nn.Conv2d(self.out_dim, self.out_dim, 3, 1, 1, bias=False),
                                      nn.BatchNorm2d(self.out_dim),
                                      nn.ReLU(True))

        self.sal_pred = nn.Sequential(nn.Conv2d(self.out_dim, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64),
                                      nn.ReLU(True),
                                      nn.Conv2d(64, 1, 3, 1, 1, bias=False))

        self.linear4 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
        self.linear3 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
        self.linear2 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)

        self.aspp_rgb = _AtrousSpatialPyramidPoolingModule(512, 320,
                                                       output_stride=16)
        self.aspp_depth = _AtrousSpatialPyramidPoolingModule(512, 320,
                                                       output_stride=16)
        self.after_aspp_conv_rgb = nn.Conv2d(320 * 5, 320, kernel_size=1, bias=False)
        self.after_aspp_conv_depth = nn.Conv2d(320 * 5, 320, kernel_size=1, bias=False)

        self.edge_conv = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.rcab_sal_edge = RCAB(32 * 2)
        self.fused_edge_sal = nn.Conv2d(64, 1, kernel_size=3, padding=1, bias=False)
        self.sal_conv = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(True)



    def forward(self, x,  feature_list, feature_list_depth):
        R1, R2, R3, R4 = feature_list[0], feature_list[1], feature_list[2], feature_list[3]
        D1, D2, D3, D4 = feature_list_depth[0], feature_list_depth[1], feature_list_depth[2], feature_list_depth[3]

        R4 = self.aspp_rgb(R4)
        D4 = self.aspp_depth(D4)
        R4 = self.after_aspp_conv_rgb(R4)
        D4 = self.after_aspp_conv_depth(D4)

        RD1 = self.fuse1(R1, D1)
        RD2 = self.fuse2(R2, D2)
        RD3 = self.fuse3(R3, D3)
        RD4 = self.fuse4(R4, D4)

        RD43 = self.up2(RD4)
        RD43 = torch.cat((RD43, RD3), dim=1)
        RD43 = self.Conv43(RD43)

        RD432 = self.up2(RD43)
        RD432 = torch.cat((RD432, RD2), dim=1)
        RD432 = self.Conv432(RD432)

        RD4321 = self.up2(RD432)
        RD4321 = torch.cat((RD4321, RD1), dim=1)
        RD4321 = self.Conv4321(RD4321)  # [B, 128, 56, 56]

        sal_map = self.sal_pred(RD4321)
        sal_out = self.up4(sal_map)

        mask4 = F.interpolate(self.linear4(RD4), size=x.size()[2:], mode='bilinear', align_corners=False)
        mask3 = F.interpolate(self.linear3(RD43), size=x.size()[2:], mode='bilinear', align_corners=False)
        mask2 = F.interpolate(self.linear4(RD432), size=x.size()[2:], mode='bilinear', align_corners=False)


        return sal_out, mask4, mask3, mask2


class PvtNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        key = []
        self.encoder_rgb = pvt_v2_b2()
        self.encoder_depth = pvt_v2_b2()
        self.decoder = Decoder(dim=128)
        self.edge_layer = Edge_Module()
        self.fuse_canny_edge = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)

        # ------------------------  rgb prediction module  ---------------------------- #
        self.conv1_1 = nn.Conv2d(512, 320, kernel_size=(3, 3), padding=(1, 1))
        self.conv1_3 = nn.Conv2d(320, 128, kernel_size=(3, 3), padding=(1, 1))
        self.conv1_5 = nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1))
        self.conv1_7 = nn.Conv2d(64, 32, kernel_size=(3, 3), padding=(1, 1))
        self.conv1_9 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))
        self.conv1_11 = nn.Conv2d(32, 1, kernel_size=(3, 3), padding=(1, 1))

        # ------------------------  t prediction module  ---------------------------- #
        self.conv2_1 = nn.Conv2d(512, 320, kernel_size=(3, 3), padding=(1, 1))
        self.conv2_3 = nn.Conv2d(320, 128, kernel_size=(3, 3), padding=(1, 1))
        self.conv2_5 = nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1))
        self.conv2_7 = nn.Conv2d(64, 32, kernel_size=(3, 3), padding=(1, 1))
        self.conv2_9 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))
        self.conv2_11 = nn.Conv2d(32, 1, kernel_size=(3, 3), padding=(1, 1))

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, input_rgb, input_depth):
        # output of backbone
        # x_size = input_rgb.size()
        rgb_feats = self.encoder_rgb(input_rgb)
        depth_feats = self.encoder_depth(input_depth)
        if self.training is True:
            # ------------------------  rgb prediction module  ---------------------------- #
            sal1 = self.conv1_11(self.upsample(self.conv1_9(self.upsample(self.conv1_7(
                self.upsample(self.conv1_5(
                    self.upsample(self.conv1_3(
                        self.upsample(self.conv1_1(rgb_feats[3])))))))))))

            # ------------------------  t prediction module  ---------------------------- #
            sal2 = self.conv2_11(self.upsample(self.conv2_9(self.upsample(self.conv2_7(
                self.upsample(self.conv2_5(
                    self.upsample(self.conv2_3(
                        self.upsample(self.conv2_1(depth_feats[3])))))))))))


            result_final, mask4, mask3, mask2 = self.decoder(input_rgb, rgb_feats, depth_feats)

            return result_final, mask4, mask3, mask2, torch.sigmoid(sal1), torch.sigmoid(sal2)
        else:
            result_final, mask4, mask3, mask2 = self.decoder(input_rgb, rgb_feats, depth_feats)
            return result_final, mask4, mask3, mask2

