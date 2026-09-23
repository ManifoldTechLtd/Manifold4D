"""GPU-accelerated batched point cloud rendering.

Replaces the CPU numpy `render_points_zbuffer` with a PyTorch implementation
that projects + splatters all target frames in a single batched call.

Typical speedup: 81 frames of 104K points goes from ~7s (CPU) to ~0.3s (GPU).
"""

import warnings
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional

from .foreground_mask import (
    subject_front_depth, silhouette_cone_leak_mask, random_subject_crop_mask,
    fit_silhouette_ellipse)

# One-shot warning flag for empty-PC short-circuit so we surface the issue
# without spamming the log on every batch.  Reset to False to re-emit.
_empty_pc_warned: bool = False


@torch.autocast(device_type="cuda", enabled=False)
@torch.no_grad()
def render_frame_softsplat(
    points_color: torch.Tensor,   # [M, 3] float, RGB in [0, 1]
    points_pos: torch.Tensor,     # [M, 3] world-space points
    cam_c2w: torch.Tensor,        # [4, 4] camera-to-world (OpenCV convention)
    K: torch.Tensor,              # [3, 3] intrinsics
    height: int,
    width: int,
    *,
    point_weight: Optional[torch.Tensor] = None,   # [M] optional per-point weight
    use_zbuffer: bool = True,
    z_tolerance: float = 0.02,
    occlusion_sharpness: float = 10.0,
    mask_threshold: float = 1e-16,
    batch_size: int = int(1e7),
) -> tuple:
    """Vista4D-style soft point-cloud splatting for ONE target view.

    Ported from ``Vista4D/utils/point_cloud/point_cloud.py::render_frame``.
    Each point is splatted to its 4 neighbouring pixels with **bilinear
    sub-pixel weights** (no integer-pixel quantisation) and colours are
    **depth-weighted soft-blended** under a **log-depth z-buffer with a
    relative tolerance** so points on the same surface merge instead of
    one hard-overwriting the others.  This removes the blocky / jittery /
    salt-and-pepper artefacts of the old square-splat last-write-wins
    rasteriser.

    Convention: ``cam_c2w`` is an OpenCV camera-to-world matrix (x right,
    y down, z forward) — same as everywhere else in manifold4d.  No axis
    flip is applied (the ``diag(-1,-1,1,1)`` flip in Vista4D's
    ``render_video`` is its external-convention adapter, NOT part of
    ``render_frame``).

    Args:
        points_color: [M, 3] float in [0, 1].
        points_pos:   [M, 3] world points.
        cam_c2w:      [4, 4] target camera-to-world.
        K:            [3, 3] intrinsics matching (height, width).
        point_weight: optional [M] extra per-point multiplicative weight
            (used by the time-alpha mode to fold in temporal opacity).
        use_zbuffer:  True  → log-depth z-buffer + depth-weighted blend
                              (z-buffer / nearest-kf modes).
                      False → order-independent weighted accumulation with
                              no depth weighting (alpha-time mode).
        z_tolerance / occlusion_sharpness: Vista4D blend hyper-params.
        mask_threshold: accumulated-weight cutoff for the validity mask.
        batch_size:   point chunk size to bound peak memory.

    Returns:
        frame_rgb:   [H, W, 3] float32 in [0, 1] (0 where no coverage).
        frame_depth: [H, W]    float32 (weighted target-cam z; 0 where empty).
        valid_mask:  [H, W]    bool.
    """
    device = points_pos.device
    num_points = points_pos.shape[0]
    HW = height * width

    accum_color = torch.zeros((HW, 3), dtype=torch.float32, device=device)
    accum_depth = torch.zeros((HW,), dtype=torch.float32, device=device)
    accum_weight = torch.zeros((HW,), dtype=torch.float32, device=device)
    z_buffer = torch.full((HW,), float("inf"), dtype=torch.float32, device=device)
    min_z_batch = torch.empty((HW,), dtype=torch.float32, device=device)

    cam_w2c = torch.linalg.inv(cam_c2w.to(torch.float32))
    R_w2c = cam_w2c[:3, :3]
    T_w2c = cam_w2c[:3, 3]
    K_f = K.to(torch.float32)
    has_weight = point_weight is not None

    if num_points == 0:
        return (torch.zeros(height, width, 3, dtype=torch.float32, device=device),
                torch.zeros(height, width, dtype=torch.float32, device=device),
                torch.zeros(height, width, dtype=torch.bool, device=device))

    for i in range(0, num_points, batch_size):
        pos_b = points_pos[i:i + batch_size].to(torch.float32)
        col_b = points_color[i:i + batch_size].to(torch.float32)
        pw_b = (point_weight[i:i + batch_size].to(torch.float32)
                if has_weight else None)

        # World -> camera, drop points behind the camera.
        cam_b = (pos_b @ R_w2c.T) + T_w2c
        depth_b = cam_b[:, 2]
        valid_z = depth_b > 1e-5
        if not valid_z.any():
            continue
        cam_b = cam_b[valid_z]; col_b = col_b[valid_z]; depth_b = depth_b[valid_z]
        if has_weight:
            pw_b = pw_b[valid_z]

        # Project to the image plane.  uvz[:, 2] == z (K's last row is [0,0,1]).
        uvz = cam_b @ K_f.T
        u = uvz[:, 0] / uvz[:, 2]
        v = uvz[:, 1] / uvz[:, 2]

        # Bilinear expansion: 1 point -> 4 sub-pixel fragments.
        u_0 = torch.floor(u).to(torch.int32)
        v_0 = torch.floor(v).to(torch.int32)
        du = u - u_0; dv = v - v_0
        w_00 = (1 - du) * (1 - dv); w_01 = (1 - du) * dv
        w_10 = du * (1 - dv);       w_11 = du * dv
        u_frag = torch.cat([u_0, u_0, u_0 + 1, u_0 + 1])
        v_frag = torch.cat([v_0, v_0 + 1, v_0, v_0 + 1])
        w_geom = torch.cat([w_00, w_01, w_10, w_11])
        d_frag = depth_b.repeat(4)
        c_frag = col_b.repeat(4, 1)
        pw_frag = pw_b.repeat(4) if has_weight else None

        if use_zbuffer:
            # Depth weighting: nearer points weigh more (soft occlusion).
            d_safe = d_frag.clamp(min=1e-1)
            w_depth = torch.pow(d_safe, -occlusion_sharpness).clamp(min=1e-10)
            w_final = w_geom * w_depth
        else:
            w_final = w_geom.clone()
        if has_weight:
            w_final = w_final * pw_frag

        valid_uv = ((u_frag >= 0) & (u_frag < width)
                    & (v_frag >= 0) & (v_frag < height) & (w_final > 0))
        if not valid_uv.any():
            continue
        idx_flat = (v_frag[valid_uv] * width + u_frag[valid_uv]).to(torch.int64)
        d_frag = d_frag[valid_uv]; c_frag = c_frag[valid_uv]; w_final = w_final[valid_uv]

        if not use_zbuffer:
            # Order-independent weighted accumulation (alpha-time mode).
            accum_color.index_add_(0, idx_flat, c_frag * w_final[..., None])
            accum_depth.index_add_(0, idx_flat, d_frag * w_final)
            accum_weight.index_add_(0, idx_flat, w_final)
            continue

        # Painter's order (far -> near) for the z-buffer logic below.
        sort_idx = torch.argsort(d_frag, descending=True)
        idx_flat = idx_flat[sort_idx]; d_frag = d_frag[sort_idx]
        c_frag = c_frag[sort_idx]; w_final = w_final[sort_idx]

        # Intra-batch occlusion culling (last write per pixel == nearest).
        min_z_batch.fill_(float("inf"))
        min_z_batch[idx_flat] = d_frag
        min_d_batch = min_z_batch[idx_flat]
        rel_diff_batch = (torch.log1p(d_frag) - torch.log1p(min_d_batch)).abs()
        visible_in_batch = rel_diff_batch <= z_tolerance
        if not visible_in_batch.any():
            continue
        idx_flat = idx_flat[visible_in_batch]; d_frag = d_frag[visible_in_batch]
        c_frag = c_frag[visible_in_batch]; w_final = w_final[visible_in_batch]

        # Global z-buffer update with log-depth tolerance: only reset a pixel
        # when a fragment is *noticeably* closer than the stored surface.
        z_curr = z_buffer[idx_flat]
        rel_diff = (torch.log1p(d_frag) - torch.log1p(z_curr)).abs()
        is_closer = d_frag < z_curr
        should_reset = is_closer & (rel_diff > z_tolerance)
        if should_reset.any():
            reset_idx = idx_flat[should_reset]
            accum_color.index_fill_(0, reset_idx, 0.0)
            accum_depth.index_fill_(0, reset_idx, 0.0)
            accum_weight.index_fill_(0, reset_idx, 0.0)
            z_buffer[reset_idx] = d_frag[should_reset]
        should_blend = should_reset | (rel_diff <= z_tolerance)
        if not should_blend.any():
            continue
        idx_blend = idx_flat[should_blend]; w_blend = w_final[should_blend]
        accum_color.index_add_(0, idx_blend, c_frag[should_blend] * w_blend[..., None])
        accum_depth.index_add_(0, idx_blend, d_frag[should_blend] * w_blend)
        accum_weight.index_add_(0, idx_blend, w_blend)

    valid_mask = accum_weight > mask_threshold
    frame_rgb = torch.zeros((HW, 3), dtype=torch.float32, device=device)
    frame_depth = torch.zeros((HW,), dtype=torch.float32, device=device)
    w_valid = accum_weight[valid_mask].clamp(min=1e-12)
    frame_rgb[valid_mask] = accum_color[valid_mask] / w_valid[..., None]
    frame_depth[valid_mask] = accum_depth[valid_mask] / w_valid
    return (frame_rgb.reshape(height, width, 3),
            frame_depth.reshape(height, width),
            valid_mask.reshape(height, width))


@torch.no_grad()
def batch_render_zbuffer(
    pts_world: torch.Tensor,       # [N, 3] float32, world-space points
    colors: torch.Tensor,          # [N, 3] float32, RGB in [0, 1]
    K: torch.Tensor,               # [3, 3] intrinsic matrix
    c2w_batch: torch.Tensor,       # [T, 4, 4] camera-to-world for T target frames
    H: int,
    W: int,
    point_radius: int = 2,         # DEPRECATED: ignored (soft splat is sub-pixel)
    extra_valid: Optional[torch.Tensor] = None,   # [T, N] bool — optional AND mask
    return_depth: bool = False,    # also emit per-frame target depth [T,H,W]
) -> tuple:
    """Render point cloud into T target camera views using the Vista4D-style
    soft splatter (:func:`render_frame_softsplat`).

    Each frame is rendered independently with bilinear sub-pixel splatting +
    depth-weighted soft blending under a log-depth z-buffer.  Replaces the
    old integer-pixel square-splat last-write-wins rasteriser.

    Args:
        pts_world:  [N, 3] world points (GPU tensor)
        colors:     [N, 3] colors in [0,1] (GPU tensor)
        K:          [3, 3] intrinsics (GPU tensor)
        c2w_batch:  [T, 4, 4] target camera poses (GPU tensor)
        H, W:       render resolution
        point_radius: DEPRECATED and ignored — kept for call-site
            compatibility.  Sub-pixel bilinear coverage replaces the old
            square-dilation hole fill.
        extra_valid: optional ``[T, N]`` bool mask selecting which points
            contribute to each target frame (e.g. dynamic-point gating).
            Geometric (depth>0 / in-bounds) validity is handled inside the
            kernel; ``None`` (default) lets every point contribute.

    Returns:
        images: [T, H, W, 3] float32, rendered RGB in [0, 1]
        masks:  [T, H, W] bool, validity masks
    """
    device = pts_world.device
    T = c2w_batch.shape[0]
    images = torch.zeros(T, H, W, 3, device=device, dtype=torch.float32)
    masks = torch.zeros(T, H, W, device=device, dtype=torch.bool)
    depths = (torch.zeros(T, H, W, device=device, dtype=torch.float32)
              if return_depth else None)

    for t in range(T):
        if extra_valid is not None:
            m = extra_valid[t]
            if not bool(m.any()):
                continue
            pts_t = pts_world[m]; col_t = colors[m]
        else:
            pts_t = pts_world; col_t = colors
        if pts_t.shape[0] == 0:
            continue
        rgb, depth, vmask = render_frame_softsplat(
            col_t, pts_t, c2w_batch[t], K, H, W, use_zbuffer=True)
        images[t] = rgb
        masks[t] = vmask
        if return_depth:
            depths[t] = depth

    if return_depth:
        return images, masks, depths
    return images, masks


# ─────────────────────────────────────────────────────────────────────────────
# Nearest-keyframe z-buffer  — sharp dynamic content via hard per-frame filter
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def batch_render_zbuffer_nearest_kf(
    pts_world: torch.Tensor,       # [N, 3]
    colors: torch.Tensor,          # [N, 3]
    K: torch.Tensor,               # [3, 3]
    c2w_batch: torch.Tensor,       # [T, 4, 4]
    pts_frame_ids: torch.Tensor,   # [N] int  — source-frame id per point
    tgt_frame_ids: torch.Tensor,   # [T] int  — frame id per target c2w
    H: int,
    W: int,
    point_radius: int = 2,
) -> tuple:
    """Z-buffer render where each target frame uses ONLY the single source
    keyframe whose frame_id is closest to the target frame_id.

    Designed to be paired with a global render (z-buffer over all keyframes)
    so the two complement each other:
      * global render     — full coverage incl. background fill-in, but
                          dynamic objects appear at K overlapping positions
      * nearest-kf render — single sharp position of dynamic objects, but
                          has coverage holes outside the active keyframe

    Rasterisation uses the Vista4D-style soft splatter
    (:func:`render_frame_softsplat`); ``point_radius`` is ignored.
    """
    device = pts_world.device
    T = c2w_batch.shape[0]
    N = pts_world.shape[0]

    images = torch.zeros(T, H, W, 3, device=device, dtype=torch.float32)
    masks = torch.zeros(T, H, W, device=device, dtype=torch.bool)

    # Defensive empty-PC short-circuit (also caught upstream in
    # ``render_video_gpu``, but keep it here so direct callers don't
    # crash either).  ``unique`` on an empty tensor returns ``[]`` which
    # makes the subsequent ``argmin(dim=1)`` choke on a zero-size dim.
    if N == 0:
        return images, masks

    pts_fids = pts_frame_ids.to(device=device, dtype=torch.float32)  # [N]
    tgt_fids = tgt_frame_ids.to(device=device, dtype=torch.float32)  # [T]

    # Per-target-frame, pick the source-keyframe id with smallest |Δt|.
    unique_pf = torch.unique(pts_fids)                                  # [K]
    dist_tk = (tgt_fids.unsqueeze(1) - unique_pf.unsqueeze(0)).abs()    # [T, K]
    nearest_fid = unique_pf[dist_tk.argmin(dim=1)]                      # [T]

    for t in range(T):
        # Hard per-frame keyframe filter: only this target frame's nearest
        # source keyframe contributes.  Geometric validity is handled inside
        # the kernel.
        m = pts_fids == nearest_fid[t]
        if not bool(m.any()):
            continue
        rgb, _, vmask = render_frame_softsplat(
            colors[m], pts_world[m], c2w_batch[t], K, H, W, use_zbuffer=True)
        images[t] = rgb
        masks[t] = vmask

    return images, masks


# ─────────────────────────────────────────────────────────────────────────────
# Foreground-occlusion ("leaked background") masking
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _foreground_leak_mask(
    depths: torch.Tensor,          # [T, rH, rW] rendered target depth (0=empty)
    pts: torch.Tensor,             # [N, 3] world points (GPU)
    cols: torch.Tensor,            # [N, 3] colors (GPU)
    is_dyn: torch.Tensor,          # [N] bool — dynamic-subject points
    pts_frame_ids: Optional[np.ndarray],
    K: torch.Tensor,               # [3, 3] intrinsics at render res
    c2ws: torch.Tensor,            # [T, 4, 4] target poses (GPU)
    src_c2ws_np: np.ndarray,       # [T, 4, 4] source poses
    fg_src_silhouette: np.ndarray, # [T,Hs,Ws] or [1,T,Hs,Ws] dilated src silhouette
    render_h: int,
    render_w: int,
    T: int,
    *,
    freeze_dynamic: bool,
    eps: float,
    depth_pct: float,
    device: torch.device,
    crop: Optional[torch.Tensor] = None,   # [N] bool — dyn pts dropped from render RGB
    freeze_frame: Optional[int] = None,    # bullet-time: single source frame of the frozen subject
) -> torch.Tensor:
    """Per-frame leaked-background mask at RENDER resolution → [T, rH, rW] bool.

    For each target frame the moving subject (``is_dyn`` points at that frame)
    is the occluder: its fitted bounding ELLIPSE (principal-axis min-enclosing
    of the source silhouette, see
    :func:`manifold4d.rendering.foreground_mask.fit_silhouette_ellipse`) + most-forward
    source depth define an infinite occlusion cone (see
    :func:`manifold4d.rendering.foreground_mask.silhouette_cone_leak_mask`).  Pixels
    that reproject into the ellipse and sit behind the subject are masked,
    EXCEPT the subject's own rendered footprint (re-rendered here so the
    dynamic content is never punched out — "don't filter the dynamic points").

    ``crop`` (augmentation): when given, the points it flags are the dynamic
    slab dropped from the render RGB.  The occlusion cone still uses the **full**
    subject (silhouette + ``z_front`` unchanged) so the slab's disocclusion is
    masked, but the footprint exemption re-renders only the **kept** points
    (``is_dyn & ~crop``) — so the cropped slab is NOT exempted and shows a clean
    hole instead of leaked background.
    """
    src_c2ws = torch.as_tensor(np.asarray(src_c2ws_np), device=device).float()
    sil = torch.as_tensor(np.asarray(fg_src_silhouette), device=device).float()
    if sil.dim() == 4:                                   # [1, T, Hs, Ws]
        sil = sil[0]
    if tuple(sil.shape[-2:]) != (render_h, render_w):
        sil = F.interpolate(sil.unsqueeze(1), size=(render_h, render_w),
                            mode="nearest").squeeze(1)
    sil = sil > 0.5                                      # [T, rH, rW] bool
    fids = (torch.as_tensor(np.asarray(pts_frame_ids), device=device)
            if pts_frame_ids is not None else None)

    leak = torch.zeros(T, render_h, render_w, dtype=torch.bool, device=device)
    # Bullet-time (freeze_dynamic): the frozen subject was unprojected from a
    # SINGLE source frame, so the occlusion cone must use THAT frame's source
    # pose + silhouette for every target frame (not the per-target index, which
    # would show the subject at a different position than the frozen cloud).
    freeze_src = (int(freeze_frame)
                  if (freeze_dynamic and freeze_frame is not None
                      and 0 <= int(freeze_frame) < src_c2ws.shape[0])
                  else None)
    for t in range(T):
        sel = (is_dyn if (freeze_dynamic or fids is None)
               else (is_dyn & (fids == t)))
        if not bool(sel.any()):
            continue
        # Source-view index for the cone: pinned to the freeze frame in
        # bullet-time, else the per-target frame (matches training).
        s = freeze_src if freeze_src is not None else t
        # Cone front + occluder use the FULL subject so a cropped slab is
        # still treated as occluder (its disocclusion gets masked).
        z_front = subject_front_depth(pts[sel], src_c2ws[s], depth_pct=depth_pct)
        if z_front is None:
            continue
        # Occluder region = the subject's fitted bounding ELLIPSE (principal-axis
        # minimum-enclosing), not the raw dilated silhouette: it closes the
        # concave gaps of a spread / limbed subject.  Fall back to the raw
        # silhouette when the fit fails (too few pixels).
        occ = fit_silhouette_ellipse(sil[s])
        if occ is None:
            occ = sil[s]
        leak_t = silhouette_cone_leak_mask(
            depths[t], K, c2ws[t], src_c2ws[s], occ, z_front, eps=eps)
        # Exempt the subject's own footprint at frame t — but only the KEPT
        # points (those actually rendered into the render RGB).  A cropped slab
        # is therefore not exempted → its leaked background is punched to a
        # hole rather than shown.
        sel_keep = sel if crop is None else (sel & ~crop)
        if bool(sel_keep.any()):
            _, _, subj_cover = render_frame_softsplat(
                cols[sel_keep], pts[sel_keep], c2ws[t], K, render_h, render_w,
                use_zbuffer=True)
        else:
            subj_cover = torch.zeros(render_h, render_w, dtype=torch.bool,
                                     device=device)
        leak[t] = leak_t & ~subj_cover
    return leak


# ─────────────────────────────────────────────────────────────────────────────
# High-level API matching prepare_multicam_data usage
# ─────────────────────────────────────────────────────────────────────────────

def render_video_gpu(
    pts_np: np.ndarray,         # [N, 3] float32
    cols_np: np.ndarray,        # [N, 3] float32 (0-1)
    c2ws_np: np.ndarray,        # [T, 4, 4] float32 target camera poses
    K_np: np.ndarray,           # [3, 3] intrinsic
    render_h: int,
    render_w: int,
    target_h: int,
    target_w: int,
    point_radius: int = 2,
    device: torch.device = None,
    pts_frame_ids: Optional[np.ndarray] = None,    # [N] int — enables dynamic current-frame gating / nearest-kf modes
    tgt_frame_ids: Optional[np.ndarray] = None,    # [T] int — default: arange(T)
    nearest_kf: bool = False,                      # True ⇒ hard nearest-keyframe z-buffer
    pts_is_dynamic: Optional[np.ndarray] = None,   # [N] bool — gate dynamic pts to nearest-kf
    return_motion_mask: bool = False,              # True ⇒ also emit per-pixel "from-dynamic-point" mask
    freeze_dynamic: bool = False,                  # True ⇒ bullet-time: render dynamic pts at EVERY target frame (no nearest-kf gate)
    foreground_masking: bool = False,              # True ⇒ punch leaked-background holes behind the dynamic subject
    src_c2ws_np: Optional[np.ndarray] = None,      # [T,4,4] per-frame SOURCE poses (required for foreground_masking)
    fg_src_silhouette: Optional[np.ndarray] = None,# [T,Hs,Ws] or [1,T,Hs,Ws] dilated source subject silhouette
    fg_eps: float = 0.02,                          # depth slack on the occlusion-cone front face
    fg_depth_pct: float = 0.0,                     # percentile of subject src-depth for the front face (0 = min)
    pts_crop_mask: Optional[np.ndarray] = None,    # [N] bool — dyn pts dropped from render RGB (augmentation)
    freeze_frame: Optional[int] = None,            # bullet-time: single source frame of the frozen subject (fg cone)
) -> tuple:
    """End-to-end GPU point-cloud rendering: numpy in, torch out.

    The default mode is classic painter's-algorithm z-buffering over all
    static points. ``nearest_kf=True`` selects a per-target-frame hard
    nearest-keyframe z-buffer and requires ``pts_frame_ids``.

    ``pts_is_dynamic`` (optional): per-point bool array marking points that
    came from dynamic-mask pixels (e.g. people). When provided in the default
    global mode, dynamic points contribute only when their source-frame index
    equals the target-frame index; static points retain global z-buffer
    coverage. Requires ``pts_frame_ids``. It has no effect in ``nearest_kf``
    mode because every point is already hard-filtered per frame.

    ``return_motion_mask`` (optional, default ``False``): when True, also
    emits a per-pixel motion mask at the target camera grid — value ``1``
    means "this pixel in the rendered video was rasterised from a point
    flagged as dynamic in ``pts_is_dynamic``" (the union of dynamic
    coverage across the same time mode used for the main render).  This
    is the render-stream "motion" channel for Vista4D-style 2-channel masks
    (alpha + motion).  When ``pts_is_dynamic`` is None or all-False, the
    motion mask is identically zero.  Implementation: the dynamic-only
    render is done as a second pass on the dynamic subset of the point
    cloud, reusing the same temporal mode (z-buffer / alpha-time /
    nearest-kf) as the main render so the motion mask aligns 1:1 with
    the rendered video.

    Renders at (render_h, render_w) then resizes to (target_h, target_w).

    Returns:
        When ``return_motion_mask=False`` (legacy):
            (render_video, render_mask)
                render_video: [3, T, target_h, target_w] float in [-1, 1]
                render_mask:  [1, T, target_h, target_w] float in {0, 1}
        When ``return_motion_mask=True``:
            (render_video, render_mask, motion_mask)
                motion_mask: [1, T, target_h, target_w] float in {0, 1}
    """
    if device is None:
        device = torch.device("cuda")

    pts = torch.from_numpy(pts_np).to(device)
    cols = torch.from_numpy(cols_np).to(device)
    K = torch.from_numpy(K_np).to(device)
    c2ws = torch.from_numpy(c2ws_np).to(device)

    # ── Empty-PC short-circuit ─────────────────────────────────────────────
    # The dataset can legitimately produce N=0 PCs (e.g. an all-sky DL3DV
    # scene with ``skip_sky=True``, or every depth pixel falling outside
    # ``[depth_min, depth_max]``).  ``batch_render_zbuffer`` and
    # ``batch_render_alpha_time`` handle this silently, but
    # ``batch_render_zbuffer_nearest_kf`` calls ``argmin`` on an empty
    # ``unique(pts_fids)`` which crashes.  Catch all three modes uniformly
    # by returning the natural "no coverage" signal: -1.0 black video
    # (VAE-encodes to an in-distribution latent — same as any sample with
    # full disocclusion) + zero mask.  Trainer's ``encode_rgb`` + mask
    # downsample handle this fine; the render branch is effectively masked
    # out at every spatial location.
    if pts.shape[0] == 0:
        global _empty_pc_warned
        if not _empty_pc_warned:
            warnings.warn(
                "render_video_gpu: received empty point cloud (N=0); "
                "returning all-black render + zero mask.  This is in-"
                "distribution but indicates a likely all-sky scene with "
                "skip_sky=True or a depth map fully outside "
                "[depth_min, depth_max].  Suppressing further warnings "
                "from this process.",
                RuntimeWarning, stacklevel=2)
            _empty_pc_warned = True
        T = c2ws.shape[0]
        render_video = -torch.ones(3, T, target_h, target_w,
                                 device=device, dtype=torch.float32)
        render_mask = torch.zeros(1, T, target_h, target_w,
                                device=device, dtype=torch.float32)
        if return_motion_mask:
            motion_mask = torch.zeros(1, T, target_h, target_w,
                                      device=device, dtype=torch.float32)
            return render_video, render_mask, motion_mask
        return render_video, render_mask

    use_nearest = bool(nearest_kf and pts_frame_ids is not None)

    # ── Dynamic-mask gating for the global render path ────────────────────────
    # Build a [T, N] AND-mask: static points pass through unchanged, dynamic
    # points are restricted to the target frame with the same source-frame id.
    # Only meaningful in the global z-buffer path; nearest-kf already applies
    # a strict per-frame filter to every point.
    # ``freeze_dynamic`` (bullet-time): the dynamic points are already a single
    # frozen instance (the dataset only unprojected them from one frame), so we
    # must NOT gate them to a single target frame — they should render at EVERY
    # target frame (frozen subject, moving camera).  Skipping the gating leaves
    # ``extra_valid`` None so the z-buffer / alpha-time path treats the frozen
    # dynamic points exactly like static content (no per-frame filter).  The
    # motion-mask second pass below still uses ``pts_is_dynamic`` to mark the
    # frozen subject's pixels at every frame.
    extra_valid = None
    # Current-frame gate for the DYNAMIC subset alone ([T, N], crop-free),
    # saved so the motion-mask second pass stays aligned with the render RGB.
    # Without it the motion pass renders EVERY dynamic source frame at every
    # target frame → a smeared "all point cloud" blob (e.g. at frame 0).
    motion_dyn_gate = None
    if (pts_is_dynamic is not None and pts_frame_ids is not None
            and not use_nearest and not freeze_dynamic):
        is_dyn = torch.from_numpy(np.asarray(pts_is_dynamic)).to(
            device=device, dtype=torch.bool)
        if bool(is_dyn.any()):
            T = c2ws.shape[0]
            fids_p_full = torch.from_numpy(
                np.asarray(pts_frame_ids)).to(device=device, dtype=torch.float32)
            if tgt_frame_ids is None:
                fids_t_full = torch.arange(T, device=device, dtype=torch.float32)
            else:
                fids_t_full = torch.from_numpy(
                    np.asarray(tgt_frame_ids)).to(device=device,
                                                  dtype=torch.float32)
            fid_match = fids_p_full.unsqueeze(0) == fids_t_full.unsqueeze(1)
            # static OR (dynamic AND at its current source frame)
            extra_valid = (~is_dyn).unsqueeze(0) | fid_match              # [T, N]
            motion_dyn_gate = fid_match                                   # [T, N]

    # ── Dynamic-crop augmentation: drop a subject slab from the render RGB ─────
    # ``pts_crop_mask`` flags dynamic points removed from the accumulated
    # GLOBAL point-cloud render so the slab shows as a hole (the motion-mask second
    # pass and the occlusion cone still see the FULL subject — see below).
    # Fold it into ``extra_valid`` (False ⇒ point excluded at every frame).
    # Skipped in nearest-kf mode (the "current" render has no global accumulation
    # to punch out).
    crop_t = None
    if pts_crop_mask is not None and not use_nearest:
        crop_t = torch.from_numpy(np.asarray(pts_crop_mask)).to(
            device=device, dtype=torch.bool)
        if bool(crop_t.any()):
            keep_row = (~crop_t).unsqueeze(0)                            # [1, N]
            if extra_valid is None:
                extra_valid = keep_row.expand(c2ws.shape[0], -1)        # [T, N]
            else:
                extra_valid = extra_valid & keep_row
        else:
            crop_t = None

    # ── Foreground (leaked-background) masking gate ─────────────────────────
    # Only meaningful for the accumulated GLOBAL z-buffer render:
    # the nearest-kf "current" render is a single source frame with no leaked
    # background to punch out.  Requires per-frame source poses + the dilated
    # source silhouette + dynamic-point flags.
    do_fg = (foreground_masking and not use_nearest
             and src_c2ws_np is not None and fg_src_silhouette is not None
             and pts_is_dynamic is not None)

    if use_nearest:
        T = c2ws.shape[0]
        fids_p = torch.from_numpy(np.asarray(pts_frame_ids)).to(device)
        if tgt_frame_ids is None:
            fids_t = torch.arange(T, device=device)
        else:
            fids_t = torch.from_numpy(np.asarray(tgt_frame_ids)).to(device)
        images, masks = batch_render_zbuffer_nearest_kf(
            pts, cols, K, c2ws, fids_p, fids_t,
            render_h, render_w, point_radius=point_radius,
        )
    else:
        out = batch_render_zbuffer(
            pts, cols, K, c2ws, render_h, render_w,
            point_radius=point_radius,
            extra_valid=extra_valid, return_depth=do_fg,
        )
        images, masks = out[0], out[1]
        depths = out[2] if do_fg else None
    # images: [T, render_h, render_w, 3], masks: [T, render_h, render_w]

    # ── Foreground masking: punch leaked-background holes behind the subject ─
    # Operates at RENDER resolution (where K matches the depth grid) BEFORE the
    # resize, folding the leak into both the validity mask (→ 0 = hole) and the
    # RGB (→ black).  Default-off; the non-fg path above is untouched.
    if do_fg:
        is_dyn_fg = torch.from_numpy(np.asarray(pts_is_dynamic)).to(
            device=device, dtype=torch.bool)
        if bool(is_dyn_fg.any()):
            leak = _foreground_leak_mask(
                depths, pts, cols, is_dyn_fg, pts_frame_ids, K, c2ws,
                src_c2ws_np, fg_src_silhouette, render_h, render_w,
                c2ws.shape[0], freeze_dynamic=freeze_dynamic, eps=fg_eps,
                depth_pct=fg_depth_pct, device=device, crop=crop_t,
                freeze_frame=freeze_frame)
            masks = masks & ~leak
            images = images.clone()
            images[leak] = 0.0          # → -1 (black) after the *2-1 below
    elif crop_t is not None and pts_is_dynamic is not None:
        # ── Crop-only hole punch (foreground_masking OFF, crop augmentation ON) ─
        # When the occlusion-cone fg path is disabled, the cropped slab points
        # were still removed from the render RGB via ``extra_valid`` — but the
        # z-buffer then fills that region with whatever points sit BEHIND it
        # (leaked background, alpha=1).  To keep the crop augmentation's intent
        # (slab → clean hole so the model inpaints the body from the motion
        # cue), punch the cropped points' OWN target footprint to a hole,
        # exempting the KEPT dynamic-subject footprint (adjacent body parts
        # stay) — the same footprint-exemption idea the fg cone uses, minus the
        # cone (no src poses / silhouette needed).
        is_dyn_c = torch.from_numpy(np.asarray(pts_is_dynamic)).to(
            device=device, dtype=torch.bool)
        fids_c = (torch.as_tensor(np.asarray(pts_frame_ids), device=device)
                  if pts_frame_ids is not None else None)
        crop_hole = torch.zeros(c2ws.shape[0], render_h, render_w,
                                dtype=torch.bool, device=device)
        for t in range(c2ws.shape[0]):
            csel = crop_t if fids_c is None else (crop_t & (fids_c == t))
            if not bool(csel.any()):
                continue
            _, _, crop_cov = render_frame_softsplat(
                cols[csel], pts[csel], c2ws[t], K, render_h, render_w,
                use_zbuffer=True)
            # Exempt the kept dynamic subject (is_dyn & ~crop) so we only punch
            # the disoccluded slab, not adjacent body parts that survive.
            ksel = ((is_dyn_c & ~crop_t) if fids_c is None
                    else (is_dyn_c & ~crop_t & (fids_c == t)))
            if bool(ksel.any()):
                _, _, keep_cov = render_frame_softsplat(
                    cols[ksel], pts[ksel], c2ws[t], K, render_h, render_w,
                    use_zbuffer=True)
                crop_hole[t] = crop_cov & ~keep_cov
            else:
                crop_hole[t] = crop_cov
        masks = masks & ~crop_hole
        images = images.clone()
        images[crop_hole] = 0.0

    # Resize to target if needed
    if render_h != target_h or render_w != target_w:
        # [T, H, W, 3] → [T, 3, H, W] for interpolate
        images_perm = images.permute(0, 3, 1, 2)  # [T, 3, rH, rW]
        images_perm = F.interpolate(images_perm, size=(target_h, target_w),
                                    mode='bilinear', align_corners=False)
        masks_f = masks.float().unsqueeze(1)  # [T, 1, rH, rW]
        masks_f = F.interpolate(masks_f, size=(target_h, target_w), mode='nearest')
        masks_resized = masks_f.squeeze(1) > 0.5  # [T, tH, tW]
    else:
        images_perm = images.permute(0, 3, 1, 2)  # [T, 3, tH, tW]
        masks_resized = masks  # [T, tH, tW]

    # Convert to training format:
    # render_video: [3, T, H, W] in [-1, 1]
    render_video = images_perm.permute(1, 0, 2, 3) * 2.0 - 1.0  # [3, T, tH, tW]
    # render_mask: [1, T, H, W] float
    render_mask = masks_resized.float().unsqueeze(0)  # [1, T, tH, tW]

    if not return_motion_mask:
        return render_video, render_mask

    # ── Motion mask: dynamic-only second render pass ─────────────────────
    # We re-render the same scene but with only the dynamic-flagged subset of
    # the point cloud, in the same z-buffer mode and under the same current-frame
    # gate (``motion_dyn_gate``)
    # as the main render.  The resulting alpha mask is exactly "this pixel was
    # rasterised from a dynamic point in THIS frame" — perfectly aligned with
    # the render_video.  Without the gate the pass would smear every dynamic
    # source frame onto every target frame (all points showing up at frame 0).
    T = c2ws.shape[0]
    if pts_is_dynamic is None:
        motion_mask = torch.zeros(1, T, target_h, target_w,
                                  device=device, dtype=torch.float32)
        return render_video, render_mask, motion_mask

    is_dyn = torch.from_numpy(np.asarray(pts_is_dynamic)).to(
        device=device, dtype=torch.bool)
    if not bool(is_dyn.any()):
        # No dynamic points → motion mask is identically zero.
        motion_mask = torch.zeros(1, T, target_h, target_w,
                                  device=device, dtype=torch.float32)
        return render_video, render_mask, motion_mask

    pts_d = pts[is_dyn]
    cols_d = cols[is_dyn]

    if use_nearest:
        # Slice frame-id alongside points; tgt_frame_ids unchanged.
        fids_p_full = torch.from_numpy(np.asarray(pts_frame_ids)).to(device)
        fids_p_d = fids_p_full[is_dyn]
        if tgt_frame_ids is None:
            fids_t_full = torch.arange(T, device=device)
        else:
            fids_t_full = torch.from_numpy(
                np.asarray(tgt_frame_ids)).to(device)
        _, masks_dyn = batch_render_zbuffer_nearest_kf(
            pts_d, cols_d, K, c2ws, fids_p_d, fids_t_full,
            render_h, render_w, point_radius=point_radius,
        )
    else:
        ev_d = (motion_dyn_gate[:, is_dyn]
                if motion_dyn_gate is not None else None)
        _, masks_dyn = batch_render_zbuffer(
            pts_d, cols_d, K, c2ws, render_h, render_w,
            point_radius=point_radius, extra_valid=ev_d,
        )
    # masks_dyn: [T, render_h, render_w] (bool or float depending on mode).

    # Resize to target (same logic as main render_mask).
    if render_h != target_h or render_w != target_w:
        mm_f = masks_dyn.float().unsqueeze(1)        # [T, 1, rH, rW]
        mm_f = F.interpolate(mm_f, size=(target_h, target_w), mode='nearest')
        motion_mask_resized = (mm_f.squeeze(1) > 0.5).float()
    else:
        motion_mask_resized = masks_dyn.float()
    motion_mask = motion_mask_resized.unsqueeze(0)   # [1, T, tH, tW]

    return render_video, render_mask, motion_mask


