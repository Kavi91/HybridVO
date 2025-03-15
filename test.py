# HybridVO/test.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.hybrid_data import HybridDataset
from models.hybrid_model import HybridModel
import yaml
import numpy as np
import wandb
from torch.cuda.amp import autocast
from tqdm import tqdm
from utils.helper import normalize_angle_delta

# Load config
config = yaml.load(open('config/config.yml'), Loader=yaml.Loader)

# Initialize W&B for evaluation
if config.get('use_wandb', False):
    wandb_mode = "online" if os.getenv("WANDB_MODE") != "offline" else "offline"
    wandb.init(
        project=config['wandb_project'],
        name='hybrid_model_evaluation',
        config=config,
        mode=wandb_mode
    )

# Dataset and Dataloader
test_seqs = config['test_seqs'].split(',')
dataset = HybridDataset(test_seqs)
dataloader = DataLoader(
    dataset,
    batch_size=config['batch_size'],
    shuffle=False,
    num_workers=config['num_workers'],
    pin_memory=True
)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridModel(device=device).to(device)
checkpoint_path = 'models/hybrid_model_best.pth'
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))
    print(f"Loaded checkpoint from {checkpoint_path}")
else:
    print(f"Warning: Checkpoint {checkpoint_path} not found. Using random initialization.")
model.eval()

# Weighted MSE loss (matching training)
class WeightedMSELoss(nn.Module):
    def __init__(self, trans_weight=1.0, rot_weight=100.0):
        super(WeightedMSELoss, self).__init__()
        self.trans_weight = trans_weight
        self.rot_weight = rot_weight

    def forward(self, pred, target):
        # Normalize rotation angles in pred and target to [-pi, pi]
        pred_rot = pred[:, :, 3:].detach().cpu().numpy()
        target_rot = target[:, :, 3:].detach().cpu().numpy()
        pred_rot = torch.tensor([[[normalize_angle_delta(angle) for angle in angles] for angles in frame] for frame in pred_rot], device=pred.device, dtype=pred.dtype)
        target_rot = torch.tensor([[[normalize_angle_delta(angle) for angle in angles] for angles in frame] for frame in target_rot], device=target.device, dtype=target.dtype)
        
        pred_normalized = torch.cat((pred[:, :, :3], pred_rot), dim=2)
        target_normalized = torch.cat((target[:, :, :3], target_rot), dim=2)
        
        trans_loss = torch.mean((pred_normalized[:, :, :3] - target_normalized[:, :, :3]) ** 2) * self.trans_weight
        rot_loss = torch.mean((pred_normalized[:, :, 3:] - target_normalized[:, :, 3:]) ** 2) * self.rot_weight
        total_loss = trans_loss + rot_loss
        return total_loss, trans_loss, rot_loss

criterion = WeightedMSELoss(trans_weight=1.0, rot_weight=100.0)

# Metrics computation
def compute_metrics(pred_poses, gt_poses):
    # Normalize rotation angles for metric computation
    pred_rot = pred_poses[:, :, 3:].detach().cpu().numpy()
    gt_rot = gt_poses[:, :, 3:].detach().cpu().numpy()
    pred_rot = np.array([[[normalize_angle_delta(angle) for angle in angles] for angles in frame] for frame in pred_rot])
    gt_rot = np.array([[[normalize_angle_delta(angle) for angle in angles] for angles in frame] for frame in gt_rot])
    
    pred_poses_normalized = torch.cat((pred_poses[:, :, :3], torch.tensor(pred_rot, device=pred_poses.device, dtype=pred_poses.dtype)), dim=2)
    gt_poses_normalized = torch.cat((gt_poses[:, :, :3], torch.tensor(gt_rot, device=gt_poses.device, dtype=gt_poses.dtype)), dim=2)
    
    pred_poses = pred_poses_normalized.detach().cpu().numpy()
    gt_poses = gt_poses_normalized.detach().cpu().numpy()
    
    pred_poses_flat = pred_poses.reshape(-1, 6)
    gt_poses_flat = gt_poses.reshape(-1, 6)
    
    pred_trans = pred_poses_flat[:, :3]
    gt_trans = gt_poses_flat[:, :3]
    pred_rot = pred_poses_flat[:, 3:]
    gt_rot = gt_poses_flat[:, 3:]
    
    rmse_trans = np.sqrt(np.mean((pred_trans - gt_trans) ** 2))
    rmse_rot = np.sqrt(np.mean((pred_rot - gt_rot) ** 2))
    
    ate_trans = np.mean(np.linalg.norm(pred_trans - gt_trans, axis=1))
    
    pred_diff = pred_poses_flat[1:] - pred_poses_flat[:-1]
    gt_diff = gt_poses_flat[1:] - gt_poses_flat[:-1]
    rpe_trans = np.mean(np.linalg.norm(pred_diff[:, :3] - gt_diff[:, :3], axis=1))
    rpe_rot = np.mean(np.linalg.norm(pred_diff[:, 3:] - gt_diff[:, 3:], axis=1))
    
    return {
        'rmse_trans': rmse_trans,
        'rmse_rot': rmse_rot,
        'ate_trans': ate_trans,
        'rpe_trans': rpe_trans,
        'rpe_rot': rpe_rot
    }

# Evaluation on test set
total_loss = 0
total_trans_loss = 0
total_rot_loss = 0
total_metrics = {'rmse_trans': 0, 'rmse_rot': 0, 'ate_trans': 0, 'rpe_trans': 0, 'rpe_rot': 0}
num_batches = len(dataloader)

# Single progress bar for the entire test evaluation
with tqdm(total=num_batches, desc="Testing", unit='batch') as pbar:
    with torch.no_grad():
        for batch_idx, (rgb, lidar, poses) in enumerate(dataloader):
            rgb, lidar, poses = rgb.to(device), lidar.to(device), poses.to(device)
            with autocast(enabled=config['use_mixed_precision']):
                outputs = model(rgb, lidar)
                loss, trans_loss, rot_loss = criterion(outputs, poses)
            total_loss += loss.item()
            total_trans_loss += trans_loss.item()
            total_rot_loss += rot_loss.item()
            
            metrics = compute_metrics(outputs, poses)
            for key in total_metrics:
                total_metrics[key] += metrics[key]
            
            pbar.update(1)
            pbar.set_postfix({'loss': loss.item(), 'grad_norm': 0.0})

# Average metrics
avg_loss = total_loss / num_batches
avg_trans_loss = total_trans_loss / num_batches
avg_rot_loss = total_rot_loss / num_batches
for key in total_metrics:
    total_metrics[key] /= num_batches

# Log to W&B
if config.get('use_wandb', False):
    wandb.log({
        'test_loss': avg_loss,
        'test_trans_loss': avg_trans_loss,
        'test_rot_loss': avg_rot_loss,
        'test_rmse_trans': total_metrics['rmse_trans'],
        'test_rmse_rot': total_metrics['rmse_rot'],
        'test_ate_trans': total_metrics['ate_trans'],
        'test_rpe_trans': total_metrics['rpe_trans'],
        'test_rpe_rot': total_metrics['rpe_rot']
    })
    wandb.finish()

# Print results
print(f"Test Metrics: Loss: {avg_loss:.4f}, Trans Loss: {avg_trans_loss:.4f}, Rot Loss: {avg_rot_loss:.4f}, "
      f"RMSE Trans: {total_metrics['rmse_trans']:.4f}, RMSE Rot: {total_metrics['rmse_rot']:.4f}, "
      f"ATE Trans: {total_metrics['ate_trans']:.4f}, RPE Trans: {total_metrics['rpe_trans']:.4f}, "
      f"RPE Rot: {total_metrics['rpe_rot']:.4f}")