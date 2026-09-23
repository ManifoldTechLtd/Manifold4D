# 3D point cloud operations: unproject depth to 3D, maintain 3D cache

import torch
import numpy as np
from typing import Optional


def make_intrinsics(focal_mm: float, sensor_mm: float, image_size: int) -> np.ndarray:
    """
    Compute 3x3 intrinsic matrix from physical camera parameters.

    Args:
        focal_mm: focal length in mm
        sensor_mm: sensor size in mm (assuming square sensor)
        image_size: image resolution (assuming square image)

    Returns:
        K: [3, 3] intrinsic matrix
    """
    fx = fy = focal_mm * image_size / sensor_mm
    cx = cy = image_size / 2.0
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ], dtype=np.float64)
    return K


def depth_to_points(
    depth: torch.Tensor,
    K: torch.Tensor,
    c2w: torch.Tensor,
    depth_mode: str = "linear",
) -> torch.Tensor:
    """
    Unproject depth map to 3D world points.

    Args:
        depth: [H, W] depth map (relative, 0-255 uint8 or 0-1 float)
        K: [3, 3] intrinsic matrix
        c2w: [4, 4] camera-to-world transform
        depth_mode: "linear" or "inverse"
            - "linear": depth_metric = depth_value (treat as linear depth)
            - "inverse": depth_metric = 1.0 / (depth_value + eps) (treat as disparity)

    Returns:
        points: [H*W, 3] world-space 3D points
    """
    H, W = depth.shape
    device = depth.device

    # Create pixel grid
    v, u = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    ones = torch.ones_like(u)
    pixels = torch.stack([u, v, ones], dim=-1)  # [H, W, 3]

    # Normalize depth
    if depth.dtype == torch.uint8:
        depth_norm = depth.float() / 255.0
    else:
        depth_norm = depth.float()

    # Convert to metric-like depth
    if depth_mode == "linear":
        depth_metric = depth_norm.clamp(min=1e-3)
    elif depth_mode == "inverse":
        depth_metric = 1.0 / (depth_norm.clamp(min=1e-3))
    else:
        raise ValueError(f"Unknown depth_mode: {depth_mode}")

    # Unproject to camera space: P_cam = K^{-1} @ pixel * depth
    K_inv = torch.inverse(K.float())  # [3, 3]
    cam_points = (pixels @ K_inv.T) * depth_metric.unsqueeze(-1)  # [H, W, 3]

    # Transform to world space: P_world = c2w @ P_cam
    cam_points_h = torch.cat([
        cam_points.reshape(-1, 3),
        torch.ones(H * W, 1, device=device)
    ], dim=-1)  # [H*W, 4]

    world_points = (c2w.float() @ cam_points_h.T).T[:, :3]  # [H*W, 3]

    return world_points


def depth_to_points_with_colors(
    depth: torch.Tensor,
    image: torch.Tensor,
    K: torch.Tensor,
    c2w: torch.Tensor,
    depth_mode: str = "linear",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Unproject depth map to colored 3D points.

    Args:
        depth: [H, W] depth map
        image: [H, W, 3] RGB image (0-255 uint8 or 0-1 float)
        K: [3, 3] intrinsic matrix
        c2w: [4, 4] camera-to-world transform
        depth_mode: "linear" or "inverse"

    Returns:
        points: [H*W, 3] world-space 3D points
        colors: [H*W, 3] RGB colors (0-1 float)
    """
    points = depth_to_points(depth, K, c2w, depth_mode=depth_mode)

    # Flatten colors
    if image.dtype == torch.uint8:
        colors = image.float().reshape(-1, 3) / 255.0
    else:
        colors = image.float().reshape(-1, 3)

    return points, colors


class PointCloudCache:
    """
    Maintains a 3D point cloud cache from multiple source frames.
    Used to accumulate observations from condition frames.
    """

    def __init__(self):
        self.points = []   # list of [N_i, 3] tensors
        self.colors = []   # list of [N_i, 3] tensors

    def add_frame(
        self,
        depth: torch.Tensor,
        image: torch.Tensor,
        K: torch.Tensor,
        c2w: torch.Tensor,
        depth_mode: str = "linear",
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Add a frame's depth + RGB to the point cloud cache.

        Args:
            depth: [H, W] depth map
            image: [H, W, 3] RGB image
            K: [3, 3] intrinsic matrix
            c2w: [4, 4] camera-to-world transform
            depth_mode: "linear" or "inverse"
            mask: [H, W] optional binary mask (1 = valid pixel to include)
        """
        pts, cols = depth_to_points_with_colors(depth, image, K, c2w, depth_mode)

        if mask is not None:
            valid = mask.reshape(-1).bool()
            pts = pts[valid]
            cols = cols[valid]

        self.points.append(pts)
        self.colors.append(cols)

    def get_all(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return all accumulated points and colors."""
        if len(self.points) == 0:
            raise RuntimeError("No frames added to cache")
        return torch.cat(self.points, dim=0), torch.cat(self.colors, dim=0)

    def clear(self):
        """Clear the cache."""
        self.points = []
        self.colors = []

    def __len__(self):
        return sum(p.shape[0] for p in self.points)
