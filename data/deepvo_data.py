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

class DeepVOData(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs
        self.seq_len = config['seq_len']
        self.img_h, self.img_w = config['img_h'], config['img_w']
        self.transform = transforms.Compose([
            transforms.Resize((self.img_h, self.img_w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config['img_means'], std=config['img_stds'])
        ])
        self.image_paths = self._get_data_info()

    def _get_data_info(self):
        X_path = []
        for folder in self.seqs:
            fpaths = glob.glob(f"{config['image_dir']}{folder}/image_2/*.png")
            fpaths.sort()
            print(f"Sequence {folder}: Found {len(fpaths)} images at {config['image_dir']}{folder}/image_2/")
            if not fpaths:
                print(f"Warning: No .png files found in {config['image_dir']}{folder}/image_2/")
                continue
            jump = self.seq_len - 1
            n_frames = len(fpaths)
            x_segs = [fpaths[i:i+self.seq_len+1] for i in range(0, n_frames - self.seq_len, jump)]
            X_path.extend(x_segs)
        return X_path

    def __getitem__(self, index):
        image_path_seq = self.image_paths[index]
        image_seq = []
        for img_path in image_path_seq:
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0)
            image_seq.append(img_tensor)
        image_seq = torch.cat(image_seq, 0)  # (seq_len+1, 3, 180, 608)
        stacked_seq = torch.cat((image_seq[:-1], image_seq[1:]), dim=1)  # (seq_len, 6, 180, 608)
        return stacked_seq

    def __len__(self):
        return len(self.image_paths)

# Test
if __name__ == '__main__':
    dataset = DeepVOData(['04'])
    img_seq = dataset[0]
    print(f"Image Sequence Shape: {img_seq.shape}")
    import matplotlib.pyplot as plt
    img = img_seq[0].permute(1, 2, 0).numpy()
    means = np.array(config['img_means'] * 2)[None, None, :]
    stds = np.array(config['img_stds'] * 2)[None, None, :]
    img = (img * stds + means).clip(0, 1)
    plt.imshow(np.concatenate((img[:, :, :3], img[:, :, 3:]), axis=1))
    plt.show()