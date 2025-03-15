# HybridVO/data/hybrid_data.py
import torch
from torch.utils.data import Dataset
import yaml
from data.deepvo_data import DeepVOData
from data.lorcon_data import LoRCoNData
import numpy as np

# Load config
config = yaml.load(open('config/config.yml'), Loader=yaml.Loader)

class HybridDataset(Dataset):
    def __init__(self, seqs):
        self.deepvo = DeepVOData(seqs)
        self.lorcon = LoRCoNData(seqs)
        self.seq_len = config['seq_len']
        assert len(self.deepvo) == len(self.lorcon), f"Mismatch: DeepVO has {len(self.deepvo)} sequences, LoRCoN has {len(self.lorcon)}"

    def __len__(self):
        return len(self.deepvo)

    def __getitem__(self, index):
        rgb_seq, rgb_poses = self.deepvo[index]  # (seq_len, 6, 184, 608), (seq_len, 6)
        lidar_seq, lidar_poses = self.lorcon[index]  # (seq_len, 10, 64, 76), (seq_len, 6)
        # Verify pose alignment (debugging)
        pose_diff = torch.abs(rgb_poses - lidar_poses).sum()
        if pose_diff > 1e-6:
            print(f"Warning: Pose mismatch at index {index}: {pose_diff.item()}")
        return rgb_seq, lidar_seq, lidar_poses  # Use LiDAR poses as ground truth for consistency

# Test
if __name__ == '__main__':
    dataset = HybridDataset(['04'])
    if len(dataset) == 0:
        print("Error: No sequences loaded.")
    else:
        rgb_seq, lidar_seq, poses = dataset[0]
        print(f"RGB Shape: {rgb_seq.shape}, LiDAR Shape: {lidar_seq.shape}, Poses Shape: {poses.shape}")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        means = np.array(config['img_means'] * 2)[None, None, :]
        stds = np.array(config['img_stds'] * 2)[None, None, :]
        rgb_img = rgb_seq[0].permute(1, 2, 0).numpy()
        rgb_img = ((rgb_img + 0.5) * stds + means).clip(0, 1)  # Reverse normalization
        ax1.imshow(np.concatenate((rgb_img[:, :, :3], rgb_img[:, :, 3:]), axis=1))
        ax1.set_title("First RGB Pair")
        ax2.imshow(lidar_seq[0, 0].numpy(), cmap='gray')
        ax2.set_title("First LiDAR Depth")
        plt.show()