# HybridVO/models/lorcon_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

# Load config
config = yaml.load(open('config/config.yml'), Loader=yaml.Loader)

class LoRCoNFeatureExtractor(nn.Module):
    def __init__(self):
        super(LoRCoNFeatureExtractor, self).__init__()
        self.simple_conv1 = nn.Conv2d(10, 64, kernel_size=3, stride=(1, 2), padding=(1, 0))  # Increased channels
        self.conv_bn1 = nn.BatchNorm2d(64)
        self.simple_conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=(1, 2), padding=(1, 0))
        self.conv_bn2 = nn.BatchNorm2d(128)
        self.simple_conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=(1, 2), padding=(1, 0))
        self.conv_bn3 = nn.BatchNorm2d(256)
        self.simple_conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=(2, 1), padding=(1, 0))
        self.conv_bn4 = nn.BatchNorm2d(512)
        self.simple_conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=(2, 1), padding=(1, 0))
        self.conv_bn5 = nn.BatchNorm2d(512)
        self.simple_conv6 = nn.Conv2d(512, 128, kernel_size=1, stride=(4, 1), padding=(0, 0))
        self.conv_bn6 = nn.BatchNorm2d(128)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=(1, 0))

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        x = x.view(batch_size * seq_len, 10, config['proj_H'], config['proj_W'])
        x = F.pad(x, (1, 1, 0, 0), mode='circular')
        x = self.maxpool(x)
        x = self.encode_image(x)
        return x

    def encode_image(self, x):
        x = F.pad(x, (1, 1, 0, 0), mode='circular')
        x = self.simple_conv1(x)
        x = self.conv_bn1(x)
        x = F.leaky_relu(x, 0.1)
        x = F.pad(x, (1, 1, 0, 0), mode='circular')
        x = self.simple_conv2(x)
        x = self.conv_bn2(x)
        x = F.leaky_relu(x, 0.1)
        x = F.pad(x, (1, 1, 0, 0), mode='circular')
        x = self.simple_conv3(x)
        x = self.conv_bn3(x)
        x = F.leaky_relu(x, 0.1)
        x = F.pad(x, (1, 1, 0, 0), mode='circular')
        x = self.simple_conv4(x)
        x = self.conv_bn4(x)
        x = F.leaky_relu(x, 0.1)
        x = F.pad(x, (1, 1, 0, 0), mode='circular')
        x = self.simple_conv5(x)
        x = self.conv_bn5(x)
        x = F.leaky_relu(x, 0.1)
        x = F.pad(x, (1, 1, 0, 0), mode='circular')
        x = self.simple_conv6(x)
        x = self.conv_bn6(x)
        x = F.leaky_relu(x, 0.1)
        target_width = 76
        if x.size(3) != target_width:
            x = F.interpolate(x, size=(x.size(2), target_width), mode='bilinear', align_corners=False)
        return x