import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import resnet34, ResNet34_Weights

# Set seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Encoder(nn.Module):
    def __init__(self, dropout_rate=0.3, leaky_relu_slope=0.01):
        super().__init__()
        # Load pretrained ResNet34
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        # Aktifkan semua parameter untuk di-train
        for param in resnet.parameters():
            param.requires_grad = True

        # Layer 1: conv1 + bn1 + activation + maxpool + layer1 + SEBlock + Dropout2d
        self.layer1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            resnet.maxpool,
            resnet.layer1,
            SEBlock(64),
            nn.Dropout2d(dropout_rate)
        )

        # Layer 2: layer2 + SEBlock + Dropout2d
        self.layer2 = nn.Sequential(
            resnet.layer2,
            SEBlock(128),
            nn.Dropout2d(dropout_rate)
        )

        # Layer 3: layer3 + SEBlock + Dropout2d
        self.layer3 = nn.Sequential(
            resnet.layer3,
            SEBlock(256),
            nn.Dropout2d(dropout_rate)
        )

        # Layer 4: layer4 + SEBlock + Dropout2d
        self.layer4 = nn.Sequential(
            resnet.layer4,
            SEBlock(512),
            nn.Dropout2d(dropout_rate)
        )

    def forward(self, x):
        x1 = self.layer1(x)  # output channels 64
        x2 = self.layer2(x1) # output channels 128
        x3 = self.layer3(x2) # output channels 256
        x4 = self.layer4(x3) # output channels 512
        return x1, x2, x3, x4
        
# Middle convolution dengan BatchNorm2d
class MiddleConv(nn.Module):
    def __init__(self, in_channels, out_channels, leaky_relu_slope=0.01):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.act1 = nn.LeakyReLU(leaky_relu_slope, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.act2 = nn.LeakyReLU(leaky_relu_slope, inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        return self.bn(x)

# Decoder block dengan BatchNorm2d
class DecBlock(nn.Module):
    def __init__(self, in_channels, out_channels, leaky_relu_slope=0.01):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(leaky_relu_slope, inplace=True)
        self.conv = nn.Conv2d(out_channels + in_channels, out_channels, 3, padding=1)
        self.se = SEBlock(out_channels)
        self.drop = nn.Dropout2d(0.5)
        self.residual_conv = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, x, skip):
        x = self.act(self.bn(self.upconv(x)))
        skip = F.interpolate(skip, size=x.size()[2:], mode='bilinear', align_corners=True)
        x_in = torch.cat([x, skip], dim=1)
        x_out = self.se(self.conv(x_in))
        x_out = self.drop(x_out)
        return self.act(x_out + self.residual_conv(x_out))  # Residual connection

# Full U-Net model dengan BatchNorm2d di final layer juga
class UNet_SE_LeakyReLU(nn.Module):
    def __init__(self, num_classes=3, dropout_rate=0.5, leaky_relu_slope=0.01):
        super().__init__()
        self.encoder = Encoder(dropout_rate, leaky_relu_slope)
        self.bridge = MiddleConv(512, 512, leaky_relu_slope)
        self.dec1 = DecBlock(512, 256, leaky_relu_slope)
        self.dec2 = DecBlock(256, 128, leaky_relu_slope)
        self.dec3 = DecBlock(128, 64, leaky_relu_slope)
        self.dec4 = DecBlock(64, 32, leaky_relu_slope)
        self.upsample = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
        self.final = nn.Sequential(
            nn.Conv2d(32 + 3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(32, num_classes, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input_skip = x
        x1, x2, x3, x4 = self.encoder(x)
        bridge = self.bridge(x4)
        d1 = self.dec1(bridge, x4)
        d2 = self.dec2(d1, x3)
        d3 = self.dec3(d2, x2)
        d4 = self.dec4(d3, x1)
        upsampled = self.upsample(d4)
        input_resized = F.interpolate(input_skip, size=(256, 256), mode='bilinear', align_corners=True)
        combined = torch.cat([upsampled, input_resized], dim=1)
        return self.final(combined)

