import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += self.shortcut(identity)
        out = F.relu(out)
        
        return out
    
class ResNet18(nn.Module):
    def __init__(self, in_channels: int = 3, n_output: int = 1000):
        super().__init__()
        self.in_channels = in_channels
        self.in_channels_block = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64,  blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, n_output)

    def _make_layer(self, out_channels, blocks, stride):
        layers = []
        layers.append(ResNetBlock(self.in_channels_block, out_channels, stride))
        self.in_channels_block = out_channels
        
        for _ in range(1, blocks):
            layers.append(ResNetBlock(out_channels, out_channels))
            
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
class ResNet34(nn.Module):
    def __init__(self, in_channels: int = 3, n_output: int = 1000):
        super().__init__()
        self.in_channels = in_channels
        self.in_channels_block = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64,  blocks=3, stride=1)
        self.layer2 = self._make_layer(128, blocks=4, stride=2)
        self.layer3 = self._make_layer(256, blocks=6, stride=2)
        self.layer4 = self._make_layer(512, blocks=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, n_output)

    def _make_layer(self, out_channels, blocks, stride):
        layers = []
        layers.append(ResNetBlock(self.in_channels_block, out_channels, stride))
        self.in_channels_block = out_channels
        
        for _ in range(1, blocks):
            layers.append(ResNetBlock(out_channels, out_channels))
            
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
class ResNetTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, output_padding=(stride-1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=stride, output_padding=(stride-1), bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(identity)
        out = F.relu(out)
        return out
    
class ResNet18Decoder(nn.Module):
    def __init__(self, latent_dim: int = 10, out_channels: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.in_channels_block = 512

        self.fc_init = nn.Linear(latent_dim, 512 * 2 * 2)

        self.layer4 = self._make_layer(256, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, blocks=2, stride=2)
        self.layer2 = self._make_layer(64,  blocks=2, stride=2)
        self.layer1 = self._make_layer(64,  blocks=2, stride=1)

        self.uptranspose = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.final_conv = nn.ConvTranspose2d(32, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

    def _make_layer(self, out_channels, blocks, stride):
        layers = []
        layers.append(ResNetTransposeBlock(self.in_channels_block, out_channels, stride))
        self.in_channels_block = out_channels
        
        for _ in range(1, blocks):
            layers.append(ResNetTransposeBlock(out_channels, out_channels))
            
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.fc_init(x)
        x = F.relu(x)
        x = x.view(-1, 512, 2, 2)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        
        x = F.relu(self.uptranspose(x))
        x = self.final_conv(x)
        return x
    
class ResNet34Decoder(nn.Module):
    def __init__(self, latent_dim: int = 10, out_channels: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.in_channels_block = 512

        self.fc_init = nn.Linear(latent_dim, 512 * 2 * 2)

        self.layer4 = self._make_layer(256, blocks=3, stride=2)
        self.layer3 = self._make_layer(128, blocks=6, stride=2)
        self.layer2 = self._make_layer(64,  blocks=4, stride=2)
        self.layer1 = self._make_layer(64,  blocks=3, stride=1)

        self.uptranspose = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.final_conv = nn.ConvTranspose2d(32, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

    def _make_layer(self, out_channels, blocks, stride):
        layers = []
        layers.append(ResNetTransposeBlock(self.in_channels_block, out_channels, stride))
        self.in_channels_block = out_channels
        
        for _ in range(1, blocks):
            layers.append(ResNetTransposeBlock(out_channels, out_channels))
            
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.fc_init(x)
        x = F.relu(x)
        x = x.view(-1, 512, 2, 2)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        
        x = F.relu(self.uptranspose(x))
        x = self.final_conv(x)
        return x