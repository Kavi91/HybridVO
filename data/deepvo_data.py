# HybridVO/data/deepvo_data.py
import os
import glob
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import yaml

# Load config
config = yaml.load(open('config/config.yml'), Loader=yaml.Loader)

def get_data_info(folder_list, seq_len_range=(5, 5), overlap=1, sample_times=1):
    X_path, Y = [], []
    for folder in folder_list:
        fpaths = glob.glob(f"{config['image_dir']}{folder}/image_2/*.png")
        fpaths.sort()
        print(f"Sequence {folder}: Found {len(fpaths)} images at {config['image_dir']}{folder}/image_2/")
        if not fpaths:
            continue
        n_frames = len(fpaths)
        jump = seq_len_range[0] - overlap
        x_segs = [fpaths[i:i+seq_len_range[0]+1] for i in range(0, n_frames - seq_len_range[0], jump)]
        X_path.extend(x_segs)
        with open(os.path.join(config['relative_pose_folder'], f"{folder}.txt"), 'r') as f:
            poses = [list(map(float, line.strip().split())) for line in f.readlines()]
            poses = [[p[4], p[3], p[5], p[0], p[1], p[2]] for p in poses]
            y_segs = [poses[i:i+seq_len_range[0]] for i in range(0, n_frames - seq_len_range[0], jump)]
            Y.extend(y_segs)
    data = {'image_path': X_path, 'pose': Y}
    df = pd.DataFrame(data, columns=['image_path', 'pose'])
    return df

class DeepVOData(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs
        self.seq_len = config['seq_len']
        self.img_h, self.img_w = config['img_h'], config['img_w']
        self.transform = transforms.Compose([
            transforms.Resize((self.img_h, self.img_w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config['img_means'], std=config['img_stds']),
            transforms.Lambda(lambda x: x - 0.5)  # Re-added explicit shift
        ])
        cache_path = f"data/cache/deepvo_train_seqs_{''.join(seqs)}_seq{self.seq_len}.pickle"
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        if os.path.isfile(cache_path):
            self.data_info = pd.read_pickle(cache_path)
        else:
            self.data_info = get_data_info(self.seqs, seq_len_range=(self.seq_len, self.seq_len), overlap=1, sample_times=1)
            self.data_info.to_pickle(cache_path)
        self.image_paths = self.data_info['image_path'].tolist()
        self.poses = self.data_info['pose'].tolist()

    def __getitem__(self, index):
        image_path_seq = self.image_paths[index]
        image_seq = []
        for img_path in image_path_seq:
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0)
            image_seq.append(img_tensor)
        image_seq = torch.cat(image_seq, 0)  # (seq_len+1, 3, 184, 608)
        stacked_seq = torch.cat((image_seq[:-1], image_seq[1:]), dim=1)  # (seq_len, 6, 184, 608)
        poses = torch.tensor(self.poses[index], dtype=torch.float32)  # (seq_len, 6)
        return stacked_seq, poses

    def __len__(self):
        return len(self.image_paths)

# Test
if __name__ == '__main__':
    dataset = DeepVOData(['04'])
    img_seq, poses = dataset[0]
    print(f"Image Sequence Shape: {img_seq.shape}, Poses Shape: {poses.shape}")
    print(f"Sample RGB Values (first 5x5 patch, channel 0): {img_seq[0, 0, :5, :5]}")
    img_raw = Image.open(dataset.image_paths[0][0]).convert('RGB')
    img_transformed = dataset.transform(img_raw)
    print(f"Raw Transformed Sample (channel 0, first 5x5): {img_transformed[0, :5, :5]}")
    import matplotlib.pyplot as plt
    img = img_seq[0].permute(1, 2, 0).numpy()
    means = np.array(config['img_means'] * 2)[None, None, :]
    stds = np.array(config['img_stds'] * 2)[None, None, :]
    img = ((img + 0.5) * stds + means).clip(0, 1)  # Reverse normalization for visualization
    plt.imshow(np.concatenate((img[:, :, :3], img[:, :, 3:]), axis=1))
    plt.title("First RGB Pair from Sequence 04")
    plt.show()