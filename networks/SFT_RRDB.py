import torch
import torch.nn as nn

class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Empirically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Empirically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x

def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # self.norm2 = torch.nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        # self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x_in):
        x = x_in
        x = self.norm1(x)
        x = x*torch.sigmoid(x)
        x = self.conv1(x)
        # x = self.norm2(x)
        # x = x*torch.sigmoid(x)
        # x = self.conv2(x)
        if self.in_channels != self.out_channels:
            x_in = self.conv_out(x_in)

        return x + x_in

class Fuse_sft_block_RRDB(nn.Module):
    def __init__(self, in_ch, out_ch, layer, num_block=1, num_grow_ch=16):
        super().__init__()
        if layer == 0:
            ratio = 1
        elif layer <= 1:
            ratio = 2
        else:
            ratio = 4
        self.layer = layer
        # print(layer, ratio, in_ch//ratio, in_ch)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch // ratio, in_ch // ratio, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(),
            # nn.Conv2d(in_ch, in_ch, kernel_size=5, stride=1, padding=2),
            # nn.LeakyReLU(),
            # nn.Conv2d(in_ch, in_ch, kernel_size=5, stride=1, padding=2),
            # nn.LeakyReLU(),
            nn.Conv2d(in_ch // ratio, in_ch, kernel_size=3, stride=1, padding=1))
        self.encode_enc_1 = ResBlock(2*in_ch, in_ch)
        # self.encode_enc_2 = make_layer(RRDB, num_block, num_feat=in_ch, num_grow_ch=num_grow_ch)
        # self.encode_enc_2 = make_layer(SELayer, num_block, channel=in_ch, reduction = 16)
        self.encode_enc_3 = ResBlock(in_ch, out_ch)

    def forward(self, enc_feat, dec_feat, w=1.0):
        # print(enc_feat.shape, dec_feat.shape, self.layer)
        # return dec_feat
        enc_feat = self.conv(enc_feat)
        enc_feat = self.encode_enc_1(torch.cat([enc_feat, dec_feat], dim=1))
        # enc_feat = self.encode_enc_2(enc_feat)
        enc_feat = self.encode_enc_3(enc_feat)
        out = w * enc_feat + dec_feat
        return out

class Fuse_sft_block_HDRev(nn.Module):
    def __init__(self, in_ch, out_ch, num_block=1, num_grow_ch=32):
        super().__init__()
        from .HDRev_Encoder import WeightBlock, ConvLayer
        self.w1 = WeightBlock(in_channels = in_ch * 2, out_channels = in_ch)
        self.w2 = WeightBlock(in_channels = in_ch * 2, out_channels = in_ch)
        self.fuse = SELayer(channel = in_ch * 2)
        self.conv = ConvLayer(in_channels = in_ch * 2, out_channels = out_ch, kernel_size = 3, padding = 1)

    def forward(self, enc_feat, enc_fea_evs, dec_feat, w=1.0):
        cat = self.fuse(torch.cat([enc_feat, dec_feat], dim = 1))
        f1 = enc_feat * self.w1(cat)
        f2 = dec_feat * self.w2(cat)
        residual = w * self.conv(torch.cat([f1, f2], dim = 1))
        out = residual + dec_feat
        return out

class SELayer(nn.Module):
    def __init__(self, channel, reduction = 16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(channel // reduction, channel, bias = False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * (1 + y.expand_as(x))


class GCT(nn.Module):

    def __init__(self, num_channels, epsilon=1e-5, mode='l2', after_relu=False):
        super(GCT, self).__init__()

        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):

        if self.mode == 'l2':
            embedding = (x.pow(2).sum((2,3), keepdim=True) + self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
            
        elif self.mode == 'l1':
            if not self.after_relu:
                _x = torch.abs(x)
            else:
                _x = x
            embedding = _x.sum((2,3), keepdim=True) * self.alpha
            norm = self.gamma / (torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon)
        else:
            print('Unknown mode!')
            sys.exit()

        gate = 1. + torch.tanh(embedding * norm + self.beta)

        return x * gate

    

class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Conv1d(1,1,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.sigmoid=nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        y=self.gap(x) #bs,c,1,1
        y=y.squeeze(-1).permute(0,2,1) #bs,1,c
        y=self.conv(y) #bs,1,c
        y=self.sigmoid(y) #bs,1,c
        y=y.permute(0,2,1).unsqueeze(-1) #bs,c,1,1
        return x*y.expand_as(x)