import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.resnet import ResNet34_Weights
from torchsummary import summary

# ===== Helper =====
def replace_relu_with_leakyrelu(module):
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, nn.LeakyReLU(negative_slope=0.01))
        else:
            replace_relu_with_leakyrelu(child)

# ===== Encoder =====
resnet = models.resnet34(weights=ResNet34_Weights["IMAGENET1K_V1"])
for name, param in resnet.named_parameters():
    if 'layer3' in name or 'layer4' in name:
        param.requires_grad = True  # allow fine-tuning


replace_relu_with_leakyrelu(resnet)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            nn.LeakyReLU(),
            resnet.layer1
        )
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x1, x2, x3, x4

# ===== MiddleConv with Dilated Conv + Dropout =====
class MiddleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.LeakyReLU()
        self.dropout = nn.Dropout2d(0.3)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=4, dilation=4)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

# ===== BasicBlock for decoder with Dropout + Residual Connection =====
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.LeakyReLU()
        self.dropout = nn.Dropout2d(0.4)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout2 = nn.Dropout2d(0.4)
        self.relu2 = nn.LeakyReLU()


        # Residual connection - hanya jika channel berbeda
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout2(out)

        # Add residual connection
        out = out + residual
        out = self.relu2(out)

        return out

# ===== Decoder Block =====
class Dec_Block(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.relu = nn.LeakyReLU()
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv = BasicBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.upconv(x)
        x = self.relu(x)
        x = self.bn(x)
        skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

# ===== MergeLayer (MINIMALIS) =====
class MergeLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, input_img):
        input_resized = F.interpolate(input_img, size=x.shape[2:], mode='bilinear', align_corners=True)
        return torch.cat([x, input_resized], dim=1)  # Output: [B, 64+3=67, H, W]

# ===== Final Model =====
class Build_UNet(nn.Module):
    def __init__(self, num_classes=3):  # Ganti sesuai jumlah kelas
        super().__init__()
        self.encoder = Encoder()
        self.mc1 = MiddleConv(512, 1024)
        self.mc2 = MiddleConv(1024, 512)
        self.decoder1 = Dec_Block(in_channels=512, skip_channels=512, out_channels=512)
        self.decoder2 = Dec_Block(512, 256, 256)
        self.decoder3 = Dec_Block(256, 128, 128)
        self.decoder4 = Dec_Block(128, 64, 64)
        self.merge = MergeLayer()
        self.segmentation = nn.Conv2d(67, num_classes, kernel_size=1)

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        mc1 = self.mc1(x4)
        mc2 = self.mc2(mc1)
        d1 = self.decoder1(mc2, x4)
        d2 = self.decoder2(d1, x3)
        d3 = self.decoder3(d2, x2)
        d4 = self.decoder4(d3, x1)
        merged = self.merge(d4, x)  # merge dengan input asli
        out = self.segmentation(merged)  # output: (B, num_classes, H, W)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Build_UNet(num_classes=3).to(device)
summary(model, (3, 256, 256))
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
