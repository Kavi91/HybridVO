# HybridVO/test.py
import torch
from torch.utils.data import DataLoader
from data.hybrid_data import HybridDataset
from models.hybrid_model import HybridModel
import yaml
import numpy as np
import wandb
from torch.cuda.amp import autocast
from tqdm import tqdm

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
model = HybridModel(device=device).to(device)  # Pass device to model constructor
checkpoint_path = 'models/hybrid_model_best.pth'
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))
    print(f"Loaded checkpoint from {checkpoint_path}")
else:
    print(f"Warning: Checkpoint {checkpoint_path} not found. Using random initialization.")
model.eval()

# Metrics computation
def compute_metrics(pred_poses, gt_poses):
    # Detach tensors from computation graph before converting to NumPy
    pred_poses = pred_poses.detach().cpu().numpy()
    gt_poses = gt_poses.detach().cpu().numpy()
    
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
total_metrics = {'rmse_trans': 0, 'rmse_rot': 0, 'ate_trans': 0, 'rpe_trans': 0, 'rpe_rot': 0}
num_batches = len(dataloader)

# Single progress bar for the entire test evaluation
with tqdm(total=num_batches, desc="Testing", unit='batch') as pbar:
    with torch.no_grad():
        for rgb, lidar, poses in dataloader:
            rgb, lidar, poses = rgb.to(device), lidar.to(device), poses.to(device)
            with autocast(enabled=config['use_mixed_precision']):
                outputs = model(rgb, lidar)
            metrics = compute_metrics(outputs, poses)
            for key in total_metrics:
                total_metrics[key] += metrics[key]
            pbar.update(1)
            pbar.set_postfix({'loss': np.mean([m['rmse_trans'] for m in [metrics] if 'rmse_trans' in m])})  # Placeholder for loss

# Average metrics
for key in total_metrics:
    total_metrics[key] /= num_batches

# Log to W&B
if config.get('use_wandb', False):
    wandb.log({
        'test_rmse_trans': total_metrics['rmse_trans'],
        'test_rmse_rot': total_metrics['rmse_rot'],
        'test_ate_trans': total_metrics['ate_trans'],
        'test_rpe_trans': total_metrics['rpe_trans'],
        'test_rpe_rot': total_metrics['rpe_rot']
    })
    wandb.finish()

# Print results
print(f"Test Metrics: RMSE Trans: {total_metrics['rmse_trans']:.4f}, RMSE Rot: {total_metrics['rmse_rot']:.4f}, "
      f"ATE Trans: {total_metrics['ate_trans']:.4f}, RPE Trans: {total_metrics['rpe_trans']:.4f}, RPE Rot: {total_metrics['rpe_rot']:.4f}")