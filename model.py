import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models
from utils import save_net,load_net
from cbam_model import ChannelAttention,SpatialAttention
from deform_conv import DeformConv2D


class ASPDNet(nn.Module):
    def __init__(self, load_weights=False):
        super(ASPDNet, self).__init__()
        self.seen = 0
        ## frontend feature extraction
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.mid_feat  = [512,512,512]
        self.frontend = make_layers(self.frontend_feat)

        '''
        # CBAM module (convolution block attention module)
        cite as "CBAM: Convolutional Block Attention Module, 2018 ECCV"
        '''
        self.planes = self.frontend_feat[-1]
        self.ca = ChannelAttention(self.planes)
        self.sa = SpatialAttention()

        '''
        dilation convolution (Spatial Pyramid Module)
        cite as "Scale Pyramid Network for Crowd Counting, 2019 WACV"
        '''
        self.conv4_3_1 = nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2)
        self.conv4_3_2 = nn.Conv2d(512, 512, kernel_size=3, padding=4, dilation=4)
        self.conv4_3_3 = nn.Conv2d(512, 512, kernel_size=3, padding=8, dilation=8)
        self.conv4_3_4 = nn.Conv2d(512, 512, kernel_size=3, padding=12, dilation=12)
        # self.conv4 = [self.conv4_3_1, self.conv4_3_2, self.conv4_3_3, self.conv4_3_4]
        self.conv5 = nn.Conv2d(2048, 512, kernel_size=1)

        '''
        convolution layers
        '''
        self.mid_end = make_layers(self.mid_feat,in_channels = 512,)

        '''
        deformable convolution network
        cite as "Deformable Convolutional Networks, 2017 ICCV"
        '''
        self.offset1 = nn.Conv2d(512, 18, kernel_size=3, padding=1)
        self.conv6_1 = DeformConv2D(512, 256, kernel_size=3, padding=1)
        self.bn6_1 = nn.BatchNorm2d(256)

        self.offset2 = nn.Conv2d(256, 18, kernel_size=3, padding=1)
        self.conv6_2 = DeformConv2D(256, 128, kernel_size=3, padding=1)
        self.bn6_2 = nn.BatchNorm2d(128)

        self.offset3 = nn.Conv2d(128, 18, kernel_size=3, padding=1)
        self.conv6_3 = DeformConv2D(128, 64, kernel_size=3, padding=1)
        self.bn6_3 = nn.BatchNorm2d(64)

        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        if not load_weights:
            mod = models.vgg16(pretrained = True)
            self._initialize_weights()
            for i in range(len(self.frontend.state_dict().items())):
                list(self.frontend.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]
    def forward(self,x):
        x = self.frontend(x)
        residual = x
        x = self.ca(x) * x
        x = self.sa(x) * x
        x += residual
        ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##
        x1 = self.conv4_3_1(x)
        x2 = self.conv4_3_2(x)
        x3 = self.conv4_3_2(x)
        x4 = self.conv4_3_2(x)
        x = torch.cat((x1, x2, x3, x4), 1)
        x = self.conv5(x)

        # x = self.backend(x)
        x = self.mid_end(x)

        offset1 = self.offset1(x)
        x = F.relu(self.conv6_1(x, offset1))
        x = self.bn6_1(x)

        offset2 = self.offset2(x)
        x = F.relu(self.conv6_2(x, offset2))
        x = self.bn6_2(x)

        offset3 = self.offset3(x)
        x = F.relu(self.conv6_3(x, offset3))
        x = self.bn6_3(x)
        ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1

    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3,padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


