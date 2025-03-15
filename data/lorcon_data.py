# HybridVO/data/lorcon_data.py
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import yaml
from utils.helper import normalize_angle_delta  # Import for normalization

# Load config
config = yaml.load(open('config/config.yml'), Loader=yaml.Loader)

def get_lidar_data_info(folder_list, seq_len_range=(5, 5), overlap=1, sample_times=1):
    X_path, Y = [], []
    for folder in folder_list:
        depth_dir = os.path.join(config['preprocessed_folder'], folder, 'depth')
        depth_paths = sorted([os.path.join(depth_dir, f) for f in os.listdir(depth_dir) if f.endswith('.npy')])
        n_frames = len(depth_paths)
        print(f"Sequence {folder}: Found {n_frames} LiDAR frames (depth, intensity, normal) at {config['preprocessed_folder']}{folder}/")
        if n_frames < seq_len_range[0] + 1:
            print(f"Warning: Sequence {folder} has too few frames ({n_frames}) for seq_len={seq_len_range[0]}")
            continue
        jump = seq_len_range[0] - overlap
        x_segs = [depth_paths[i:i+seq_len_range[0]+1] for i in range(0, n_frames - seq_len_range[0], jump)]
        X_path.extend(x_segs)
        with open(os.path.join(config['relative_pose_folder'], f"{folder}.txt"), 'r') as f:
            poses = [list(map(float, line.strip().split())) for line in f.readlines()]
            poses = [[p[4], p[3], p[5], p[0], p[1], p[2]] for p in poses]  # Reorder to [yaw, pitch, roll, x, y, z]
            # Normalize rotation angles
            for pose in poses:
                pose[0] = normalize_angle_delta(pose[0])  # yaw
                pose[1] = normalize_angle_delta(pose[1])  # pitch
                pose[2] = normalize_angle_delta(pose[2])  # roll
            y_segs = [poses[i:i+seq_len_range[0]] for i in range(0, n_frames - seq_len_range[0], jump)]
            Y.extend(y_segs)
    data = {'image_path': X_path, 'pose': Y}
    df = pd.DataFrame(data, columns=['image_path', 'pose'])
    return df

class LoRCoNData(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs
        self.seq_len = config['seq_len']
        self.proj_H, self.proj_W = config['proj_H'], 76
        cache_path = f"data/cache/lorcon_train_seqs_{''.join(seqs)}_seq{self.seq_len}.pickle"
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        if os.path.isfile(cache_path):
            self.data_info = pd.read_pickle(cache_path)
        else:
            self.data_info = get_lidar_data_info(self.seqs, seq_len_range=(self.seq_len, self.seq_len), overlap=1, sample_times=1)
            self.data_info.to_pickle(cache_path)
        self.image_paths = self.data_info['image_path'].tolist()
        self.poses = self.data_info['pose'].tolist()

    def __getitem__(self, index):
        depth_path_seq = self.image_paths[index]
        lidar_seq = []
        for i, dpath in enumerate(depth_path_seq):
            depth = np.load(dpath) / 255.0
            depth = depth[:, :76]
            intensity = np.load(dpath.replace('depth', 'intensity')) / 255.0
            intensity = intensity[:, :76]
            normal = (np.load(dpath.replace('depth', 'normal')) + 1.0) / 2.0
            normal = normal[:, :76, :]
            frame = np.concatenate([depth[None], intensity[None], normal.transpose(2, 0, 1)], axis=0)
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
                frame = np.concatenate([frame, next_frame], axis=0)
            else:
                frame = np.concatenate([frame, np.zeros_like(frame[:, :, :76])], axis=0)
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
        print("Error: No sequences loaded.")
    else:
        lidar_seq, poses = dataset[0]
        print(f"LiDAR Sequence Shape: {lidar_seq.shape}, Poses Shape: {poses.shape}")
        import matplotlib.pyplot as plt
        plt.imshow(lidar_seq[0, 0].numpy(), cmap='gray')
        plt.title("First Depth Frame from Sequence 04")
        plt.show()