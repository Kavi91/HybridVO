# HybridVO/models/hybrid_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import numpy as np
from torch.utils.checkpoint import checkpoint

# Load config
config = yaml.load(open('config/config.yml'), Loader=yaml.Loader)

class HybridModel(nn.Module):
    def __init__(self, device=None):
        super(HybridModel, self).__init__()
        from models.deepvo_model import DeepVOFeatureExtractor
        from models.lorcon_model import LoRCoNFeatureExtractor
        self.rgb_extractor = DeepVOFeatureExtractor()
        self.lidar_extractor = LoRCoNFeatureExtractor()
        
        self.resize_rgb = nn.Conv2d(1024, 1024, kernel_size=(1, 10), stride=(1, 10), padding=0)
        self.upscale_rgb = nn.ConvTranspose2d(1024, 1024, kernel_size=(1, 76), stride=(1, 76), padding=0)
        
        self.fusion_conv = nn.Conv2d(1024 + 128, 512, kernel_size=3, padding=1)
        
        self.rnn = nn.LSTM(
            input_size=512 * 4 * 76,
            hidden_size=config.get('rnn_hidden_size', 1000),
            num_layers=config.get('rnn_num_layers', 2),
            dropout=config.get('rnn_dropout_between', 0),
            batch_first=True
        )
        self.rnn_drop_out = nn.Dropout(config.get('rnn_dropout_out', 0.5))
        
        self.fc = nn.Linear(config.get('rnn_hidden_size', 1000), 6)
        
        # Store the device
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, rgb, lidar):
        rgb_features = self.rgb_extractor(rgb)
        lidar_features = self.lidar_extractor(lidar)
        
        self.rgb_features = rgb_features
        self.lidar_features = lidar_features
        
        # Normalize features before fusion
        rgb_features = (rgb_features - rgb_features.mean(dim=(0, 2, 3), keepdim=True)) / (rgb_features.std(dim=(0, 2, 3), keepdim=True) + 1e-8)
        lidar_features = (lidar_features - lidar_features.mean(dim=(0, 2, 3), keepdim=True)) / (lidar_features.std(dim=(0, 2, 3), keepdim=True) + 1e-8)
        
        rgb_features = self.resize_rgb(rgb_features)
        rgb_features = self.upscale_rgb(rgb_features)
        
        if rgb_features.size(2) != lidar_features.size(2):
            rgb_features = F.interpolate(rgb_features, size=(lidar_features.size(2), lidar_features.size(3)), mode='bilinear', align_corners=False)
        
        fused_features = torch.cat((rgb_features, lidar_features), dim=1)
        fused_features = self.fusion_conv(fused_features)
        
        self.fused_features = fused_features
        
        batch_size = rgb.size(0)
        seq_len = rgb.size(1)
        rnn_input = fused_features.view(batch_size, seq_len, -1)
        
        # Initial hidden and cell states on the model's device
        h0 = torch.zeros(config.get('rnn_num_layers', 2), batch_size, config.get('rnn_hidden_size', 1000)).to(self.device)
        c0 = torch.zeros(config.get('rnn_num_layers', 2), batch_size, config.get('rnn_hidden_size', 1000)).to(self.device)
        
        # Define a wrapper function for checkpointing
        def rnn_forward(input_tensor, hidden):
            return self.rnn(input_tensor, hidden)
        
        # Apply checkpointing to the RNN forward pass
        if config.get('use_mixed_precision', False):
            out, (h, c) = checkpoint(rnn_forward, rnn_input, (h0, c0))
        else:
            out, (h, c) = self.rnn(rnn_input, (h0, c0))
        
        out = self.rnn_drop_out(out)
        out = self.fc(out)
        return out

    def visualize_features_on_rgb_grid(self, rgb_image):
        rgb_img = rgb_image[0, 0, :3].permute(1, 2, 0).cpu().numpy()
        means = np.array(config['img_means'])[None, None, :]
        stds = np.array(config['img_stds'])[None, None, :]
        rgb_img = (rgb_img * stds + means).clip(0, 1)
        
        rgb_features = self.rgb_features[0].mean(dim=0).detach().cpu().numpy()
        lidar_features = self.lidar_features[0].mean(dim=0).detach().cpu().numpy()
        fused_features = self.fused_features[0].mean(dim=0).detach().cpu().numpy()
        
        rgb_features = torch.tensor(rgb_features).unsqueeze(0).unsqueeze(0)
        rgb_features = F.interpolate(rgb_features, size=(4, 76), mode='bilinear', align_corners=False).squeeze().numpy()
        
        rgb_features = (rgb_features - rgb_features.min()) / (rgb_features.max() - rgb_features.min() + 1e-8)
        lidar_features = (lidar_features - lidar_features.min()) / (lidar_features.max() - lidar_features.min() + 1e-8)
        fused_features = (fused_features - fused_features.min()) / (fused_features.max() - fused_features.min() + 1e-8)
        
        rgb_features = np.power(rgb_features, 0.5)
        lidar_features = np.power(lidar_features, 0.5)
        fused_features = np.power(fused_features, 0.5)
        
        rgb_features = torch.tensor(rgb_features).unsqueeze(0).unsqueeze(0)
        lidar_features = torch.tensor(lidar_features).unsqueeze(0).unsqueeze(0)
        fused_features = torch.tensor(fused_features).unsqueeze(0).unsqueeze(0)
        
        rgb_features = F.interpolate(rgb_features, size=(184, 608), mode='bilinear', align_corners=False).squeeze().numpy()
        lidar_features = F.interpolate(lidar_features, size=(184, 608), mode='bilinear', align_corners=False).squeeze().numpy()
        fused_features = F.interpolate(fused_features, size=(184, 608), mode='bilinear', align_corners=False).squeeze().numpy()
        
        overlay_rgb = rgb_img.copy()
        overlay_lidar = rgb_img.copy()
        overlay_fused = rgb_img.copy()
        alpha = 0.3
        
        overlay_rgb[:, :, 0] += alpha * rgb_features
        overlay_lidar[:, :, 2] += alpha * lidar_features
        overlay_fused[:, :, 1] += alpha * fused_features
        
        overlay_rgb = np.clip(overlay_rgb, 0, 1)
        overlay_lidar = np.clip(overlay_lidar, 0, 1)
        overlay_fused = np.clip(overlay_fused, 0, 1)
        
        plt.figure(figsize=(18, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(overlay_rgb)
        plt.title("RGB Image with DeepVO Overlay (Red)")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(overlay_lidar)
        plt.title("RGB Image with LoRCoN-LO Overlay (Blue)")
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(overlay_fused)
        plt.title("RGB Image with Fused Overlay (Green)")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()