# HybridVO/data/preprocess.py
import os
import glob
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import yaml

# Load config
config = yaml.load(open('config/config.yml'), Loader=yaml.Loader)

def calculate_rgb_mean_std(image_dir, train_seqs, img_h, img_w, minus_point_5=False):
    """Calculate mean and std for RGB channels across training dataset."""
    image_path_list = []
    for seq in train_seqs:
        image_path_list += glob.glob(f"{image_dir}{seq}/image_2/*.png")
    
    n_images = len(image_path_list)
    cnt_pixels = 0
    print(f"Number of frames in training dataset: {n_images}")
    
    to_tensor = transforms.ToTensor()
    resize = transforms.Resize((img_h, img_w))
    
    mean_tensor = [0, 0, 0]
    for idx, img_path in enumerate(image_path_list):
        print(f"{idx+1}/{n_images}", end='\r')
        img = Image.open(img_path).convert('RGB')
        img = resize(img)
        img_tensor = to_tensor(img)
        if minus_point_5:
            img_tensor = img_tensor - 0.5
        for c in range(3):
            mean_tensor[c] += float(torch.sum(img_tensor[c]))
        cnt_pixels += img_tensor.shape[1] * img_tensor.shape[2]
    
    mean_tensor = [v / cnt_pixels for v in mean_tensor]
    print(f"Mean: {mean_tensor}")
    
    std_tensor = [0, 0, 0]
    for idx, img_path in enumerate(image_path_list):
        print(f"{idx+1}/{n_images}", end='\r')
        img = Image.open(img_path).convert('RGB')
        img = resize(img)
        img_tensor = to_tensor(img)
        if minus_point_5:
            img_tensor = img_tensor - 0.5
        for c in range(3):
            std_tensor[c] += float(torch.sum((img_tensor[c] - mean_tensor[c]) ** 2))
    
    std_tensor = [np.sqrt(v / cnt_pixels) for v in std_tensor]
    print(f"Std: {std_tensor}")
    
    return mean_tensor, std_tensor

if __name__ == '__main__':
    train_seqs = config['train_seqs'].split(',')
    mean, std = calculate_rgb_mean_std(config['image_dir'], train_seqs, config['img_h'], config['img_w'], minus_point_5=True)
    print(f"Calculated Mean: {mean}, Std: {std}")