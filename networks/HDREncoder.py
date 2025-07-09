import torch
from utils import BGR2YCbCr, Gaussian_filtering
from .submodules import *

class Conv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding=1, activation='relu', norm=None) :
        super(Conv2DBlock, self).__init__()
        self.conv = nn.Sequential(
            ConvLayer(in_channels, out_channels, kernel_size, stride, padding, activation, norm),
            ConvLayer(out_channels, out_channels, kernel_size, stride, padding, activation, norm)
        )
    def forward(self, x):
        return self.conv(x)

class Conv2DBlock1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding=1, activation='relu', norm=None) :
        super(Conv2DBlock1, self).__init__()
        self.conv = nn.Sequential(
            ConvLayer(in_channels, out_channels, kernel_size, stride, padding, activation, norm)
        )
    def forward(self, x):
        return self.conv(x)

class WeightBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride=1, downsample=None, norm=None):
        super(WeightBlock, self).__init__()
        bias = False if norm == 'BN' else True
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1, bias=bias)
        self.norm = norm
        if norm == 'BN':
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.bn1 = nn.InstanceNorm2d(out_channels)
            self.bn2 = nn.InstanceNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1, bias=bias)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        if self.norm in ['BN', 'IN']:
            out = self.bn1(out)
        out = self.conv2(out)
        if self.norm in ['BN', 'IN']:
            out = self.bn2(out)
        return self.act(out)

class UEncoder4Recurrent(nn.Module):
    def __init__(self, in_channels=3):
        super(UEncoder4Recurrent, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.Maxpool1 = RecurrentConvLayer(filters[0], filters[0], kernel_size=5, padding=2, stride=2, recurrent_block_type='convlstm')#nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = RecurrentConvLayer(filters[1], filters[1], kernel_size=5, padding=2, stride=2, recurrent_block_type='convlstm')#nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = RecurrentConvLayer(filters[2], filters[2], kernel_size=5, padding=2, stride=2, recurrent_block_type='convlstm')#nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = Conv2DBlock(in_channels, filters[0], norm='IN')
        self.Conv2 = Conv2DBlock(filters[0], filters[1], norm='IN')
        self.Conv3 = Conv2DBlock(filters[1], filters[2], norm='IN')
        self.Conv4 = Conv2DBlock(filters[2], filters[3], norm='IN')


    def forward(self, x, prev_states):

        e1 = self.Conv1(x)
        if prev_states is None:
            prev_states = [None] * 4

        e2, state2 = self.Maxpool1(e1, prev_states[0])
        e2 = self.Conv2(e2)

        e3, state3 = self.Maxpool2(e2, prev_states[1])
        e3 = self.Conv3(e3)
        
        e4, state4 = self.Maxpool3(e3, prev_states[2])
        e4 = self.Conv4(e4)

        return [e1, e2, e3, e4], [state2, state3, state4]

class UEncoder4_small(nn.Module):
    def __init__(self, in_channels=3):
        super(UEncoder4_small, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.Maxpool1 = nn.Conv2d(filters[0], filters[0], kernel_size=5, padding=2, stride=2)#nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.Conv2d(filters[1], filters[1], kernel_size=5, padding=2, stride=2)#nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.Conv2d(filters[2], filters[2], kernel_size=5, padding=2, stride=2)#nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = Conv2DBlock1(in_channels, filters[0], norm='IN')
        self.Conv2 = Conv2DBlock1(filters[0], filters[1], norm='IN')
        self.Conv3 = Conv2DBlock1(filters[1], filters[2], norm='IN')
        self.Conv4 = Conv2DBlock1(filters[2], filters[3], norm='IN')


    def forward(self, x, prev_states):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)
        
        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        return [e1, e2, e3, e4], prev_states


class UDecoder4(nn.Module):
    def __init__(self, out_channels=3):
        super(UDecoder4, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Up4 = UpsampleConvLayer(filters[3], filters[2], kernel_size=5, padding=2, norm='IN')
        self.Up_conv4 = Conv2DBlock(filters[3], filters[2], norm='IN')

        self.Up3 = UpsampleConvLayer(filters[2], filters[1], kernel_size=5, padding=2, norm='IN')
        self.Up_conv3 = Conv2DBlock(filters[2], filters[1], norm='IN')

        self.Up2 = UpsampleConvLayer(filters[1], filters[0], kernel_size=5, padding=2, norm='IN')
        self.Up_conv2 = Conv2DBlock(filters[1], filters[0], norm='IN')

        self.Conv = nn.Conv2d(filters[0], out_channels, kernel_size=1, stride=1, padding=0)
        self.act = nn.Sigmoid()


    def forward(self, e):
        e1, e2, e3, e4 = e[0], e[1], e[2], e[3]

        d4 = self.Up4(e4, e3.shape[2:])
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4, e2.shape[2:])
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3, e1.shape[2:])
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        return self.act(out)

class FusionLayer_SFT(nn.Module):
    def __init__(self, n_features):
        super(FusionLayer_SFT, self).__init__()
        
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.n_features = n_features
        self.weight = nn.ModuleList()
        for i in range(n_features):
            self.weight.append(WeightBlock(in_channels = filters[i] * 2, out_channels = filters[i]))
        
    def forward(self, fe, fi):
        if self.n_features != len(fe) or self.n_features != len(fi):
            print("The number of features does not match the input features of FusionLayer")
        out = []
        for i in range(self.n_features):
            cat = torch.cat([fe[i], fi[i]], dim = 1)
            f1 = fe[i] * self.weight[i](cat)
        
            out.append(fi[i] + f1)
        return out

class FusionLayer_Unet(nn.Module):
    def __init__(self, n_features):
        super(FusionLayer_Unet, self).__init__()
        
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.n_features = n_features
        self.w1 = nn.ModuleList()
        self.w2 = nn.ModuleList()
        self.conv = nn.ModuleList()
        self.fuse = nn.ModuleList()
        for i in range(n_features):
            self.w1.append(WeightBlock(in_channels = filters[i] * 2, out_channels = filters[i]))
            self.w2.append(WeightBlock(in_channels = filters[i] * 2, out_channels = filters[i]))
            self.fuse.append(SELayer(channel = filters[i] * 2))
            self.conv.append(ConvLayer(in_channels = filters[i] * 2, out_channels = filters[i], kernel_size = 3, padding = 1))
        
    def forward(self, fe, fi):
        if self.n_features != len(fe) or self.n_features != len(fi):
            print("The number of features does not match the input features of FusionLayer")
        out = []
        for i in range(self.n_features):
            cat = self.fuse[i](torch.cat([fe[i], fi[i]], dim = 1))
            f1 = fe[i] * self.w1[i](cat)
            f2 = fi[i] * self.w2[i](cat)
        
            out.append(self.conv[i](torch.cat([f1, f2], dim = 1)))
        return out

class HDRev_Encoder(nn.Module):
    def __init__(self, 
                num_bins=5, 
                pretrained_I_path='pretrained/HDRev/Encoder_I.pth', 
                pretrained_E_path='pretrained/HDRev/Encoder_E.pth',
                pretrained_merge_path='pretrained/HDRev/Merge.pth',
                pretrained_D_path='pretrained/HDRev/Decoder.pth'):
        super(HDRev_Encoder, self).__init__()
        self.netEncoder_I = UEncoder4Recurrent(in_channels=3)
        self.netEncoder_I.load_state_dict(torch.load(pretrained_I_path, weights_only=True))

        self.netEncoder_E = UEncoder4Recurrent(in_channels=num_bins)
        self.netEncoder_E.load_state_dict(torch.load(pretrained_E_path, weights_only=True))

        self.netMerge = FusionLayer_Unet(n_features=4)
        self.netMerge.load_state_dict(torch.load(pretrained_merge_path, weights_only=True))

        self.netDecoder = UDecoder4(out_channels=3)
        self.netDecoder.load_state_dict(torch.load(pretrained_D_path, weights_only=True))

        self.statei, self.statee = None, None

        self.dtype = torch.float32

    def forward(self, ldr, evs, return_img=False):
        ldr = torch.flip(ldr, dims=[1])
        feat_i, self.statei = self.netEncoder_I(ldr, self.statei)
        feat_e, self.statee = self.netEncoder_E(evs, self.statee)

        feat = self.netMerge(feat_e, feat_i)
        # feat = [torch.cat([a, b], axis=1) for a, b in zip(feat_i, feat_e)]
        self.statei, self.statee = None, None
        if return_img:
            return feat[-1], feat, torch.flip(self.netDecoder(feat), dims=[1])
        return feat[-1], feat
