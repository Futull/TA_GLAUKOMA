import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, padding=3):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gelu1 = nn.GELU()
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.dw_conv7x7 = DepthwiseSeparableConv(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv1x1_1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.gelu2 = nn.GELU()

        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv3x3(x)
        x = self.gelu1(x)
        x = self.bn1(x)

        x = self.dw_conv7x7(x)
        x = self.bn2(x)

        x = self.conv1x1_1(x)
        x = self.conv1x1_2(x)
        x = self.gelu2(x)

        skip = x
        x_down = self.downsample(x)
        return x_down, skip


class Bottleneck(nn.Module):
    def __init__(self, channels):
        super(Bottleneck, self).__init__()
        self.dw_conv7x7 = DepthwiseSeparableConv(channels, channels)
        self.bn = nn.BatchNorm2d(channels)
        self.conv1x1_1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(channels, channels, kernel_size=1)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        x = self.dw_conv7x7(x)
        x = self.bn(x)
        x = self.conv1x1_1(x)
        x = self.conv1x1_2(x)
        x = self.gelu(x)
        x = self.dropout(x)
        return x



class AttentionModule(nn.Module):
    def __init__(self, channels):
        super(AttentionModule, self).__init__()
        self.W1 = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.W2 = nn.Conv2d(channels, channels, kernel_size=1)
        self.W3 = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x1, x2):
        if x1.size() != x2.size():
            x1 = F.interpolate(x1, size=x2.size()[2:], mode='bilinear', align_corners=True)

        c = self.W1(x1)
        c1, c2, c3 = torch.chunk(c, 3, dim=1)

        s = self.W2(c1 + x2)
        y1 = torch.sigmoid(s) * x2
        y2 = torch.sigmoid(c2) * torch.tanh(c3)
        y = self.W3(y1 + y2)

        out = y + x1
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, do_upsample=True):
        super(DecoderBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) if do_upsample else nn.Identity()
        self.conv3x3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gelu1 = nn.GELU()
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.dw_conv7x7 = DepthwiseSeparableConv(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv1x1_1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.gelu2 = nn.GELU()

        self.align = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.align_skip = nn.Conv2d(skip_channels, out_channels, kernel_size=1) if skip_channels != out_channels else nn.Identity()

        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x, skip=None):
        x = self.upsample(x)
        x = self.align(x)
        if skip is not None:
            skip = self.align_skip(skip)
            x = x + skip

        x = self.conv3x3(x)
        x = self.gelu1(x)
        x = self.bn1(x)

        x = self.dw_conv7x7(x)
        x = self.bn2(x)

        x = self.conv1x1_1(x)
        x = self.conv1x1_2(x)
        x = self.gelu2(x)
        x = self.dropout(x)
        return x


class AETUnet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=64):
        super(AETUnet, self).__init__()

        self.initial_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.initial_gelu = nn.GELU()
        self.initial_bn = nn.BatchNorm2d(base_channels)

        self.encoder1 = EncoderBlock(base_channels, base_channels)
        self.encoder2 = EncoderBlock(base_channels, base_channels * 2)
        self.encoder3 = EncoderBlock(base_channels * 2, base_channels * 4)

        self.bottleneck = Bottleneck(base_channels * 4)

        self.attention1 = AttentionModule(base_channels * 4)
        self.attention2 = AttentionModule(base_channels * 2)
        self.attention3 = AttentionModule(base_channels)

        self.decoder1 = DecoderBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.decoder2 = DecoderBlock(base_channels * 2, base_channels, base_channels)
        self.decoder3 = DecoderBlock(base_channels, base_channels, base_channels, do_upsample=False)

        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.initial_gelu(x)
        x = self.initial_bn(x)

        x1, skip1 = self.encoder1(x)
        x2, skip2 = self.encoder2(x1)
        x3, skip3 = self.encoder3(x2)

        x = self.bottleneck(x3)

        x = self.decoder1(self.attention1(x, skip3), skip2)
        x = self.decoder2(self.attention2(x, skip2), skip1)
        x = self.decoder3(self.attention3(x, skip1))

        output = self.final_conv(x)
        return output
