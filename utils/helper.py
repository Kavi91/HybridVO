# HybridVO/utils/helper.py
import numpy as np

def normalize_angle_delta(angle):
    """Normalize angle to [-pi, pi] range."""
    if angle > np.pi:
        angle = angle - 2 * np.pi
    elif angle < -np.pi:
        angle = 2 * np.pi + angle
    return angle