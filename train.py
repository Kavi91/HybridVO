# HybridVO/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data.hybrid_data import HybridDataset
from models.hybrid_model import HybridModel
import yaml
import os
import wandb
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
from utils.helper import normalize_angle_delta

# Load config
config = yaml.load(open('config/config.yml'), Loader=yaml.Loader)

# Convert weight_decay to float to avoid TypeError
weight_decay = float(config['weight_decay']) if isinstance(config['weight_decay'], str) else config['weight_decay']

# Initialize W&B if enabled
if config.get('use_wandb', False):
    wandb.init(
        project=config['wandb_project'],
        name=config['wandb_run_name'],
        config=config
    )

# Dataset and Dataloader (split into train and validation)
train_seqs = config['train_seqs'].split(',')
val_seqs = config['val_seqs'].split(',')
train_dataset = HybridDataset(train_seqs)
val_dataset = HybridDataset(val_seqs)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=config['batch_size'],
    shuffle=True,
    num_workers=config['num_workers'],
    pin_memory=True
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=config['batch_size'],
    shuffle=False,
    num_workers=config['num_workers'],
    pin_memory=True
)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridModel(device=device).to(device)

# Weighted MSE loss to prioritize rotation with angle normalization
class WeightedMSELoss(nn.Module):
    def __init__(self, trans_weight=1.0, rot_weight=100.0):
        super(WeightedMSELoss, self).__init__()
        self.trans_weight = trans_weight
        self.rot_weight = rot_weight

    def forward(self, pred, target):
        # Normalize predicted and target rotation angles to [-pi, pi] in-place
        pred_rot = pred[:, :, 3:].clone()
        target_rot = target[:, :, 3:].clone()
        for b in range(pred_rot.size(0)):
            for t in range(pred_rot.size(1)):
                pred_rot[b, t, 0] = torch.tensor(normalize_angle_delta(pred_rot[b, t, 0].item()), device=device)  # yaw
                pred_rot[b, t, 1] = torch.tensor(normalize_angle_delta(pred_rot[b, t, 1].item()), device=device)  # pitch
                pred_rot[b, t, 2] = torch.tensor(normalize_angle_delta(pred_rot[b, t, 2].item()), device=device)  # roll
                target_rot[b, t, 0] = torch.tensor(normalize_angle_delta(target_rot[b, t, 0].item()), device=device)  # yaw
                target_rot[b, t, 1] = torch.tensor(normalize_angle_delta(target_rot[b, t, 1].item()), device=device)  # pitch
                target_rot[b, t, 2] = torch.tensor(normalize_angle_delta(target_rot[b, t, 2].item()), device=device)  # roll
        
        pred_normalized = torch.cat((pred[:, :, :3], pred_rot), dim=2)
        target_normalized = torch.cat((target[:, :, :3], target_rot), dim=2)
        
        trans_loss = torch.mean((pred_normalized[:, :, :3] - target_normalized[:, :, :3]) ** 2) * self.trans_weight
        rot_loss = torch.mean((pred_normalized[:, :, 3:] - target_normalized[:, :, 3:]) ** 2) * self.rot_weight
        total_loss = trans_loss + rot_loss
        return total_loss, trans_loss, rot_loss

criterion = WeightedMSELoss(trans_weight=1.0, rot_weight=100.0)
optimizer = optim.Adam(
    model.parameters(),
    lr=config['learning_rate'],
    weight_decay=weight_decay
)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, verbose=True)

# Mixed precision setup
scaler = GradScaler(enabled=config['use_mixed_precision'])

# Training Loop
num_epochs = config['num_epochs']
best_val_loss = float('inf')
patience = config['early_stopping_patience']
patience_counter = 0

# Ensure model directory exists for saving
os.makedirs('models', exist_ok=True)

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

# Debug print to check script version
print("Using single progress bar per epoch implementation - Version 14")

# Debug print to check forward pass completion (commented out as requested)
debug_count = 0
# if debug_count % 10 == 0:
#     print(f"Processed batch {debug_count}")

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0
    train_trans_loss = 0
    train_rot_loss = 0
    train_metrics = {'rmse_trans': 0, 'rmse_rot': 0, 'ate_trans': 0, 'rpe_trans': 0, 'rpe_rot': 0}
    num_batches = len(train_dataloader)
    
    pbar = tqdm(total=num_batches, desc=f'Epoch {epoch+1}/{num_epochs} (Training)', unit='batch')
    for batch_idx, (rgb, lidar, poses) in enumerate(train_dataloader):
        rgb, lidar, poses = rgb.to(device), lidar.to(device), poses.to(device)
        optimizer.zero_grad()
        
        with autocast(enabled=config['use_mixed_precision']):
            outputs = model(rgb, lidar)
            loss, trans_loss, rot_loss = criterion(outputs, poses)
        
        scaler.scale(loss).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Increased max_norm
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item()
        train_trans_loss += trans_loss.item()
        train_rot_loss += rot_loss.item()
        
        metrics = compute_metrics(outputs, poses)
        for key in train_metrics:
            train_metrics[key] += metrics[key]
        
        pbar.update(1)
        pbar.set_postfix({'loss': loss.item(), 'grad_norm': grad_norm.item()})
        
        debug_count += 1
        # Commented out as requested
        # if debug_count % 10 == 0:
        #     print(f"Processed batch {debug_count}")
    
    pbar.close()
    
    avg_train_loss = train_loss / num_batches
    avg_train_trans_loss = train_trans_loss / num_batches
    avg_train_rot_loss = train_rot_loss / num_batches
    for key in train_metrics:
        train_metrics[key] /= num_batches
    
    # Validation
    model.eval()
    val_loss = 0
    val_trans_loss = 0
    val_rot_loss = 0
    val_metrics = {'rmse_trans': 0, 'rmse_rot': 0, 'ate_trans': 0, 'rpe_trans': 0, 'rpe_rot': 0}
    num_val_batches = len(val_dataloader)
    
    pbar = tqdm(total=num_val_batches, desc=f'Epoch {epoch+1}/{num_epochs} (Validation)', unit='batch')
    with torch.no_grad():
        for batch_idx, (rgb, lidar, poses) in enumerate(val_dataloader):
            rgb, lidar, poses = rgb.to(device), lidar.to(device), poses.to(device)
            with autocast(enabled=config['use_mixed_precision']):
                outputs = model(rgb, lidar)
                loss, trans_loss, rot_loss = criterion(outputs, poses)
            val_loss += loss.item()
            val_trans_loss += trans_loss.item()
            val_rot_loss += rot_loss.item()
            
            metrics = compute_metrics(outputs, poses)
            for key in val_metrics:
                val_metrics[key] += metrics[key]
            
            pbar.update(1)
            pbar.set_postfix({'loss': loss.item(), 'grad_norm': 0.0})
    
    pbar.close()
    
    avg_val_loss = val_loss / num_val_batches
    avg_val_trans_loss = val_trans_loss / num_val_batches
    avg_val_rot_loss = val_rot_loss / num_val_batches
    for key in val_metrics:
        val_metrics[key] /= num_val_batches
    
    # Log to W&B
    if config.get('use_wandb', True):
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'train_trans_loss': avg_train_trans_loss,
            'train_rot_loss': avg_train_rot_loss,
            'val_trans_loss': avg_val_trans_loss,
            'val_rot_loss': avg_val_rot_loss,
            'train_rmse_trans': train_metrics['rmse_trans'],
            'train_rmse_rot': train_metrics['rmse_rot'],
            'train_ate_trans': train_metrics['ate_trans'],
            'train_rpe_trans': train_metrics['rpe_trans'],
            'train_rpe_rot': train_metrics['rpe_rot'],
            'val_rmse_trans': val_metrics['rmse_trans'],
            'val_rmse_rot': val_metrics['rmse_rot'],
            'val_ate_trans': val_metrics['ate_trans'],
            'val_rpe_trans': val_metrics['rpe_trans'],
            'val_rpe_rot': val_metrics['rpe_rot'],
            'train_grad_norm': grad_norm.item() if grad_norm is not None else 0.0
        })
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
          f"Train Trans Loss: {avg_train_trans_loss:.4f}, Train Rot Loss: {avg_train_rot_loss:.4f}, "
          f"Val Trans Loss: {avg_val_trans_loss:.4f}, Val Rot Loss: {avg_val_rot_loss:.4f}, "
          f"Val RMSE Trans: {val_metrics['rmse_trans']:.4f}, Val ATE Trans: {val_metrics['ate_trans']:.4f}, "
          f"Train Grad Norm: {grad_norm.item():.4f}")
    
    # Update learning rate scheduler
    scheduler.step(avg_val_loss)
    
    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'models/hybrid_model_best.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Save final model
torch.save(model.state_dict(), 'models/hybrid_model_final.pth')

# Finalize W&B run
if config.get('use_wandb', True):
    wandb.finish()