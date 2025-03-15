# HybridVO/data/lorcon_data.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import yaml

# Load config
config = yaml.load(open('config/config.yml'), Loader=yaml.Loader)

class LoRCoNData(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs
        self.seq_len = config['seq_len']
        self.proj_H, self.proj_W = config['proj_H'], 76  # Adjusted to 76
        self.image_paths, self.poses = self._get_data_info()

    def _get_data_info(self):
        X_path = []
        Y = []
        for seq in self.seqs:
            depth_dir = os.path.join(config['preprocessed_folder'], seq, 'depth')
            depth_paths = sorted([os.path.join(depth_dir, f) 
                                for f in os.listdir(depth_dir) 
                                if f.endswith('.npy')])
            n_frames = len(depth_paths)
            # Verify intensity and normal files
            intensity_dir = os.path.join(config['preprocessed_folder'], seq, 'intensity')
            normal_dir = os.path.join(config['preprocessed_folder'], seq, 'normal')
            if len(os.listdir(intensity_dir)) != n_frames or len(os.listdir(normal_dir)) != n_frames:
                print(f"Warning: Mismatch in number of files for Sequence {seq}. Depth: {n_frames}, Intensity: {len(os.listdir(intensity_dir))}, Normal: {len(os.listdir(normal_dir))}")
            print(f"Sequence {seq}: Found {n_frames} LiDAR frames (depth, intensity, normal) at {config['preprocessed_folder']}{seq}/")
            
            if n_frames < self.seq_len + 1:
                print(f"Warning: Sequence {seq} has too few frames ({n_frames}) for seq_len={self.seq_len}")
                continue
            jump = self.seq_len - 1
            x_segs = [depth_paths[i:i+self.seq_len+1] for i in range(0, n_frames - self.seq_len, jump)]
            X_path.extend(x_segs)
            with open(os.path.join(config['relative_pose_folder'], f"{seq}.txt"), 'r') as f:
                poses = [list(map(float, line.strip().split())) for line in f.readlines()]
                poses = [[p[4], p[3], p[5], p[0], p[1], p[2]] for p in poses]
                y_segs = [poses[i:i+self.seq_len] for i in range(0, n_frames - self.seq_len, jump)]
                Y.extend(y_segs)
        print(f"Total LiDAR sequences: {len(X_path)}")
        return X_path, Y

    def __getitem__(self, index):
        depth_path_seq = self.image_paths[index]
        lidar_seq = []
        for i, dpath in enumerate(depth_path_seq):
            depth = np.load(dpath) / 255.0  # (64, 900)
            depth = depth[:, :76]  # Crop to 76
            intensity = np.load(dpath.replace('depth', 'intensity')) / 255.0  # (64, 900)
            intensity = intensity[:, :76]
            normal = (np.load(dpath.replace('depth', 'normal')) + 1.0) / 2.0  # (64, 900, 3)
            normal = normal[:, :76, :]
            frame = np.concatenate([depth[None], intensity[None], normal.transpose(2, 0, 1)], axis=0)  # (5, 64, 76)
            if i < len(depth_path_seq) - 1:
                next_dpath = depth_path_seq[i + 1]
                next_depth = np.load(next_dpath) / 255.0
                next_depth = next_depth[:, :76]
                next_intensity = np.load(next_dpath.replace('depth', 'intensity')) / 255.0
                next_intensity = next_intensity[:, :76]
                next_normal = (np.load(next_dpath.replace('depth', 'normal')) + 1.0) / 2.0
                next_normal = next_normal[:, :76, :]
                if next_normal.shape == (64, 76, 3):
                    next_normal = next_normal.transpose(2, 0, 1)
                next_frame = np.concatenate([next_depth[None], next_intensity[None], next_normal], axis=0)
                frame = np.concatenate([frame, next_frame], axis=0)  # (10, 64, 76)
            else:
                frame = np.concatenate([frame, np.zeros_like(frame[:, :, :76])], axis=0)  # Pad to 10
            lidar_seq.append(torch.tensor(frame, dtype=torch.float32))
        lidar_seq = torch.stack(lidar_seq[:-1])  # (seq_len, 10, 64, 76)
        poses = torch.tensor(self.poses[index], dtype=torch.float32)
        return lidar_seq, poses

    def __len__(self):
        return len(self.image_paths)

# Test
if __name__ == '__main__':
    dataset = LoRCoNData(['04'])
    if len(dataset) == 0:
        print("Error: No sequences loaded. Check paths and sequence data.")
    else:
        lidar_seq, poses = dataset[0]
        print(f"LiDAR Sequence Shape: {lidar_seq.shape}, Poses Shape: {poses.shape}")
        import matplotlib.pyplot as plt
        plt.imshow(lidar_seq[0, 0].numpy(), cmap='gray')
        plt.title("First Depth Frame from Sequence 04")
        plt.show()