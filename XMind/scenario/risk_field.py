"""Driver risk field (DRF) of one vehicle, following the RESPOND implementation (torus-shaped
Gaussian along the predicted arc, truncated at the look-ahead distance)."""

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


GRID_RESOLUTION = 1.0   # m
LOOK_AHEAD_TIME = 4.0   # s


def look_ahead_distance(speed: float) -> float:
    """max(4 * speed, 1) m, as in RESPOND."""
    if not np.isfinite(speed):
        raise ValueError("DRF requires a finite, non-negative speed")
    return max(LOOK_AHEAD_TIME * max(speed, 0.0), 1.0)


def compute_risk_field(
    x: float,
    y: float,
    speed: float,
    heading_rad: float,
    common_grid: Tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """DRF intensity of a vehicle sampled on `common_grid` (meshgrid in world coordinates)."""
    if not np.isfinite([x, y, speed, heading_rad]).all():
        raise ValueError("Vehicle state contains non-finite values")

    distance = look_ahead_distance(speed)
    heading = float(heading_rad) % (2.0 * np.pi)

    # Near-zero steering angle: the arc is almost a straight line along the heading.
    delta = np.deg2rad(0.1) / 54.0
    radius = abs(5.0 / np.tan(delta))
    center_angle = heading + np.sign(delta) * np.pi / 2.0
    center_x = x + radius * np.cos(center_angle)
    center_y = y + radius * np.sin(center_angle)

    # Field on the vehicle's own 1 m grid.
    step = GRID_RESOLUTION
    native_x, native_y = np.meshgrid(
        np.arange(x - 2 * distance, x + 2 * distance + step, step),
        np.arange(y - 2 * distance, y + 2 * distance + step, step),
    )

    # Arc length along the turning circle (Gaussian_3d_torus_arclen).
    ux, uy = x - center_x, y - center_y
    wx, wy = native_x - center_x, native_y - center_y
    cos_angle = (ux * wx + uy * wy) / (np.hypot(ux, uy) * np.hypot(wx, wy))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    signed_angle = np.sign(delta) * np.sign(ux * wy - wx * uy) * angle
    arc_length = radius * np.remainder(2.0 * np.pi + signed_angle, 2.0 * np.pi)

    # Amplitude, truncated at the look-ahead distance (Gaussian_3d_torus_a).
    amplitude = 0.0128 * (arc_length - distance) ** 2
    amplitude *= (
        (np.sign(distance - arc_length) + 1.0) / 2.0
        * (np.sign(amplitude) + 1.0) / 2.0
        * (np.sign(arc_length) + 1.0) / 2.0
    )

    # Lateral spread inside / outside the circle (Gaussian_3d_torus_sigma).
    sigma_inside = (0.07 + 0.5 * abs(delta)) * arc_length + 1.0
    sigma_outside = (0.07 + 2.5 * abs(delta)) * arc_length + 1.0

    # Intensity (Gaussian_3d_torus_z).
    radial_offset = np.hypot(wx, wy) - radius
    inside_weight = (1.0 - np.sign(radial_offset)) / 2.0
    outside_weight = (1.0 + np.sign(radial_offset)) / 2.0
    native_field = amplitude * (
        inside_weight * np.exp(-radial_offset ** 2 / (2 * sigma_inside ** 2))
        + outside_weight * np.exp(-radial_offset ** 2 / (2 * sigma_outside ** 2))
    )

    # Bilinear resampling onto the common grid.
    source = torch.as_tensor(native_field, dtype=torch.float32)[None, None, :, :]
    grid_x, grid_y = common_grid
    target_x = torch.as_tensor(grid_x, dtype=torch.float32)
    target_y = torch.as_tensor(grid_y, dtype=torch.float32)
    sample_x = 2 * (target_x - native_x.min()) / (native_x.max() - native_x.min()) - 1
    sample_y = 2 * (target_y - native_y.min()) / (native_y.max() - native_y.min()) - 1
    sample_grid = torch.stack((sample_x, sample_y), dim=-1)[None, :, :, :]
    result = F.grid_sample(source, sample_grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return result[0, 0].cpu().numpy()
