# HybridVO/models/deepvo_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

# Load config
config = yaml.load(open('config/config.yml'), Loader=yaml.Loader)

def conv(batch_norm, in_planes, out_planes, kernel_size=3, stride=1, dropout=0, padding=None):
    if padding is None:
        padding = (kernel_size - 1) // 2
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Dropout(dropout)
    )

class DeepVOFeatureExtractor(nn.Module):
    def __init__(self):
        super(DeepVOFeatureExtractor, self).__init__()
        self.batch_norm = config['batch_norm']
        self.conv_dropout = config['conv_dropout']
        self.conv1   = conv(self.batch_norm,   6,   64, kernel_size=7, stride=2, dropout=self.conv_dropout[0])
        self.conv2   = conv(self.batch_norm,  64,  128, kernel_size=5, stride=2, dropout=self.conv_dropout[1])
        self.conv3   = conv(self.batch_norm, 128,  256, kernel_size=5, stride=2, dropout=self.conv_dropout[2])
        self.conv3_1 = conv(self.batch_norm, 256,  256, kernel_size=3, stride=1, dropout=self.conv_dropout[3])
        self.conv4   = conv(self.batch_norm, 256,  512, kernel_size=3, stride=2, dropout=self.conv_dropout[4])
        self.conv4_1 = conv(self.batch_norm, 512,  512, kernel_size=3, stride=1, dropout=self.conv_dropout[5])
        self.conv5   = conv(self.batch_norm, 512,  512, kernel_size=3, stride=2, dropout=self.conv_dropout[6])
        self.conv5_1 = conv(self.batch_norm, 512,  512, kernel_size=3, stride=1, dropout=self.conv_dropout[7])
        self.conv6   = conv(self.batch_norm, 512, 1024, kernel_size=3, stride=2, dropout=self.conv_dropout[8])  # Original stride

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        x = x.view(batch_size * seq_len, 6, config['img_h'], config['img_w'])
        #print(f"Input: {x.shape}")
        out_conv2 = self.conv2(self.conv1(x)); #print(f"conv2: {out_conv2.shape}")
        out_conv3 = self.conv3_1(self.conv3(out_conv2)); #print(f"conv3_1: {out_conv3.shape}")
        out_conv4 = self.conv4_1(self.conv4(out_conv3)); #print(f"conv4_1: {out_conv4.shape}")
        out_conv5 = self.conv5_1(self.conv5(out_conv4)); #print(f"conv5_1: {out_conv5.shape}")
        out_conv6 = self.conv6(out_conv5); #print(f"conv6: {out_conv6.shape}")
        return out_conv6

# Test
if __name__ == '__main__':
    model = DeepVOFeatureExtractor()
    dummy_rgb = torch.randn(2, 5, 6, config['img_h'], config['img_w'])
    features = model(dummy_rgb)
    print(f"Final Feature Shape: {features.shape}")
    import matplotlib.pyplot as plt
    feat_map = features[0].mean(dim=0).detach().numpy()
    plt.imshow(feat_map, cmap='viridis')
    plt.title("Mean Feature Map from DeepVO CNN")
    plt.show()